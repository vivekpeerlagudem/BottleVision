"""Multi-object tracking.

Responsibility: give detections a persistent identity across frames. The
detector and post-processor are memoryless -- they describe a single instant.
This module holds the only state that spans time.

Algorithm: constant-velocity prediction + two-stage association (the SORT
structure, with a hand-rolled motion model instead of a Kalman filter).
Each frame:

    0. PREDICT -- shift every track's box by its estimated velocity, so we
                  match against where the bottle *is now*, not where it *was*.
    1. MATCH-1 -- IoU between each PREDICTED box and each new detection.
    2. MATCH-2 -- for whatever stage 1 missed, fall back to centre distance.
                  IoU collapses to exactly 0.0 once boxes stop overlapping, so
                  no IoU threshold can rescue fast motion; distance degrades
                  gracefully and can.
    3. UPDATE  -- matched tracks adopt the observed box and refine velocity.
    4. AGE     -- unmatched tracks get older by one frame.
    5. SPAWN   -- unmatched detections become new tracks with fresh IDs.
    6. RETIRE  -- tracks unseen longer than ``max_lost_frames`` are dropped.
    7. REPORT  -- only tracks confirmed in THIS frame are returned.

A lost track keeps coasting on its last velocity (damped), so a bottle that is
briefly occluded is re-acquired at its *predicted* position rather than its
stale one. Lost tracks are still not drawn -- their position is an estimate.

KNOWN LIMIT: this tracker is purely geometric. If two bottles physically
exchange places (especially while touching or occluding each other), nothing in
the geometry distinguishes them and their IDs may swap. Solving that requires
appearance-based re-identification, not a better motion model.

DEBUG: run ``python scripts/run_webcam.py --debug`` for a per-frame dump of
timing, predicted boxes, the IoU matrix, and which stage produced each match.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

from bottlevision.config import TrackerConfig
from bottlevision.detection import Detection, Track
from bottlevision.geometry import BBox, center, containment, iou, mean_side
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)

# Re-exported for callers that import these from this module.
__all__ = ["Tracker", "center", "iou", "mean_side"]

# How many frames of pipeline state to retain so that, when an identity event
# happens, we can show what led up to it. The decisive evidence is usually one
# frame BEFORE the visible ID change: a duplicate detection spawns a twin track,
# and the twin steals the identity on the following frame.
_EVENT_LOOKBACK_FRAMES = 6

# How far back to remember an identity, so that a bottle which went undetected
# for a while and returned with a new number is still recognised as a switch
# rather than as a new object. ~9 seconds at 10 FPS.
_IDENTITY_LOOKBACK_FRAMES = 90

# How many nearest tracks to report when a new ID has to be created.
_SPAWN_DIAGNOSTIC_TRACKS = 3

# A track that has been observed only once has no velocity yet, so prediction
# cannot help it and its gate must absorb the object's ENTIRE displacement
# between frames. Established tracks get the tight gate because prediction has
# already moved them most of the way. Without this widening a fast-moving object
# can never reach its second observation, so it never earns a velocity, so it
# never predicts -- a deadlock that spawns a fresh ID on every frame.
_BOOTSTRAP_GATE_MULTIPLIER = 3.0

# --- Assignment analysis (diagnostic only; does not affect matching) --------
# A unified cost so the greedy assignment the tracker actually makes can be
# compared against the global optimum on one scale.
#   stage-1 (IoU) pairs cost  1 - iou           -> range [0.0, 1 - threshold]
#   stage-2 (distance) pairs  1 + distance/gate -> range [1.0, 2.0]
# Every stage-2 cost therefore exceeds every stage-1 cost, preserving the
# tracker's own stage priority. Leaving a slot unmatched costs more than any
# real pair, so an assignment that matches MORE objects always wins first.
_ASSIGNMENT_PENALTY = 3.0
# Exact optimum is found by permutation search; refuse above this size.
_MAX_EXACT_ASSIGNMENT = 8


@dataclass
class _FrameRecord:
    """One frame of pipeline state, retained for identity-event forensics."""

    frame: int
    detections: list[str] = field(default_factory=list)
    duplicate_notes: list[str] = field(default_factory=list)
    before: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    reported: list[int] = field(default_factory=list)
    spawned: list[int] = field(default_factory=list)
    cost_matrix: list[str] = field(default_factory=list)
    assignment_report: list[str] = field(default_factory=list)
    greedy_suboptimal: bool = False
    # track_id -> what happened to it during association this frame
    outcomes: dict[int, str] = field(default_factory=dict)
    # track_id -> its predicted box at association time
    predicted_boxes: dict[int, BBox] = field(default_factory=dict)


@dataclass
class _ActiveTrack:
    """Internal, mutable bookkeeping for one tracked object.

    This is the tracker's private state; the public output type is
    :class:`~bottlevision.detection.Track`.

    Attributes:
        bbox: Current position estimate -- the observed box after a match, or
            the velocity-predicted box while unmatched.
        vx, vy: Estimated centre velocity in pixels per frame.
        hits: How many times this track has been observed. Velocity is only
            trusted (and prediction only applied) from the second hit onward.
        last_observed_center: Centre of the last *observed* box, used to
            measure velocity across however many frames have elapsed.
    """

    track_id: int
    bbox: BBox
    confidence: float
    class_name: str
    frames_since_seen: int = 0
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    last_observed_center: tuple[float, float] | None = None
    # --- lifecycle forensics (diagnostic only) ---
    created_frame: int = 0
    first_reported_frame: int | None = None
    last_observed_frame: int = 0
    last_observed_bbox: BBox | None = None
    history: list[str] = field(default_factory=list)


class Tracker:
    """Assigns persistent integer IDs to detections across frames."""

    def __init__(self, config: TrackerConfig) -> None:
        """Start with no tracks and the first ID equal to 1.

        Args:
            config: Validated tracker settings.
        """
        self._iou_threshold = config.iou_threshold
        self._max_center_distance_factor = config.max_center_distance_factor
        self._velocity_smoothing = config.velocity_smoothing
        self._max_lost_frames = config.max_lost_frames
        self._tracks: list[_ActiveTrack] = []
        self._next_id = 1
        # Diagnostics only.
        self._frame_number = 0
        self._last_update_time: float | None = None
        self._history: deque[_FrameRecord] = deque(maxlen=_EVENT_LOOKBACK_FRAMES)
        # What was on screen last frame: track_id -> box. Used to notice that a
        # visible object's ID changed, which is the failure that matters.
        self._previous_reported: dict[int, BBox] = {}
        # Every identity ever shown: track_id -> (last frame reported, last box).
        # Lets us recognise that a bottle which vanished for a while and came
        # back with a new number used to be a different ID.
        self._identity_ledger: dict[int, tuple[int, BBox]] = {}
        # Lifecycle histories of deleted tracks, so a switch can be explained
        # even when the original track no longer exists.
        self._graveyard: dict[int, list[str]] = {}
        _log.info(
            "Tracker ready (iou_threshold=%.2f, max_center_distance_factor=%.2f, "
            "velocity_smoothing=%.2f, max_lost_frames=%d).",
            self._iou_threshold,
            self._max_center_distance_factor,
            self._velocity_smoothing,
            self._max_lost_frames,
        )

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: This frame's detections (already filtered to bottles).

        Returns:
            The tracks confirmed in this frame, each with a stable
            ``track_id``. Tracks that are currently lost are omitted.
        """
        self._frame_number += 1
        debug = _log.isEnabledFor(logging.DEBUG)
        record = _FrameRecord(frame=self._frame_number)

        # 0. Move every track to where we expect it to be now.
        self._predict()

        if debug:
            self._log_frame_header(detections)

        # Snapshot the inputs and the PREDICTED track state, i.e. exactly what
        # association is about to work from.
        record.detections = [
            f"det{i} {d.bbox} conf={d.confidence:.2f}"
            for i, d in enumerate(detections)
        ]
        record.duplicate_notes = self._duplicate_notes(detections)
        record.before = self._render_tracks(as_list=True)

        matched_pairs, matched_detections, stages = self._associate(detections)
        matched_tracks = {track_index for track_index, _ in matched_pairs}
        record.matches = [
            f"#{self._tracks[t].track_id} <- det{d} ({stages[(t, d)]})"
            for t, d in matched_pairs
        ]
        # Capture the full association picture BEFORE _absorb overwrites the
        # predicted boxes with observations.
        self._record_association(record, detections, matched_pairs, stages)
        self._analyse_assignment(record, detections, matched_pairs)
        if record.greedy_suboptimal:
            _log.info(
                "\n%s",
                "\n".join(
                    ["", "=" * 78,
                     f"GREEDY SUBOPTIMAL at frame {record.frame}", "=" * 78]
                    + record.assignment_report
                    + ["=" * 78]
                ),
            )

        if debug:
            _log.debug(
                "  MATCHED: %s",
                [
                    f"#{self._tracks[t].track_id} <- det{d} ({stages[(t, d)]})"
                    for t, d in matched_pairs
                ]
                or "NONE",
            )

        # 3. Matched tracks adopt their new observation and refine velocity.
        for track_index, detection_index in matched_pairs:
            track = self._tracks[track_index]
            predicted = track.bbox
            self._absorb(track, detections[detection_index])
            track.last_observed_frame = self._frame_number
            track.last_observed_bbox = detections[detection_index].bbox
            self._note(
                track,
                f"TRACK MATCHED det{detection_index} via "
                f"{stages[(track_index, detection_index)]}; predicted={predicted} "
                f"observed={detections[detection_index].bbox} "
                f"hits={track.hits} v=({track.vx:+.1f},{track.vy:+.1f})",
            )

        # 4. Tracks with no match this frame grow older.
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.frames_since_seen += 1
                self._note(
                    track,
                    f"TRACK MISSED -> AGED to {track.frames_since_seen}/"
                    f"{self._max_lost_frames}; "
                    f"{record.outcomes.get(track.track_id, 'no outcome recorded')}",
                )

        # 5. Detections with no match are new objects.
        spawned: list[int] = []
        existing_before_spawn = list(self._tracks)
        claimed_by = {t_idx: d_idx for t_idx, d_idx in matched_pairs}
        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                if existing_before_spawn:
                    self._log_spawn_reason(
                        self._next_id,
                        detection,
                        detection_index,
                        existing_before_spawn,
                        claimed_by,
                    )
                new_track = _ActiveTrack(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    last_observed_center=center(detection.bbox),
                    created_frame=self._frame_number,
                    last_observed_frame=self._frame_number,
                    last_observed_bbox=detection.bbox,
                )
                self._note(
                    new_track,
                    f"TRACK CREATED from det{detection_index} {detection.bbox} "
                    f"- no existing track matched this detection "
                    f"({len(existing_before_spawn)} track(s) existed)",
                )
                self._tracks.append(new_track)
                spawned.append(self._next_id)
                self._next_id += 1

        # 6. Retire tracks that have been unseen for too long.
        surviving, expired = [], []
        for track in self._tracks:
            (surviving if track.frames_since_seen <= self._max_lost_frames
             else expired).append(track)
        for track in expired:
            self._note(
                track,
                f"TRACK DELETED - unseen for {track.frames_since_seen} frames, "
                f"exceeding max_lost_frames={self._max_lost_frames}",
            )
            self._graveyard[track.track_id] = track.history
        if len(self._graveyard) > 40:  # keep the graveyard bounded
            for stale in sorted(self._graveyard)[: len(self._graveyard) - 40]:
                del self._graveyard[stale]
        self._tracks = surviving
        retired = sorted(t.track_id for t in expired)

        if debug:
            _log.debug("  SPAWNED: %s", spawned or "none")
            _log.debug("  RETIRED: %s", retired or "none")
            _log.debug("  tracks AFTER (%d): %s", len(self._tracks), self._render_tracks())

        # 7. Report only what was actually seen in this frame.
        reported = [
            Track(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                class_name=track.class_name,
            )
            for track in self._tracks
            if track.frames_since_seen == 0
        ]

        record.after = self._render_tracks(as_list=True)
        record.reported = [t.track_id for t in reported]
        record.spawned = spawned
        self._history.append(record)

        # THE failure that matters: an object that was on screen last frame is
        # still on screen now, but under a different ID.
        current_boxes = {t.track_id: t.bbox for t in reported}
        for track in self._tracks:
            if track.frames_since_seen == 0 and track.first_reported_frame is None:
                track.first_reported_frame = self._frame_number
                self._note(
                    track,
                    "TRACK CONFIRMED (note: this tracker has NO confirmation "
                    "stage - a track is shown on its very first hit)",
                )

        self._log_roster(record, current_boxes)
        self._report_identity_switches(record, current_boxes, detections, matched_pairs)
        self._report_identity_handovers(record, current_boxes)

        for track_id, box in current_boxes.items():
            self._identity_ledger[track_id] = (self._frame_number, box)
        self._previous_reported = current_boxes

        return reported

    def _report_identity_handovers(
        self, record: _FrameRecord, current_boxes: dict[int, BBox]
    ) -> None:
        """Detect and explain a visible object changing from one ID to another.

        A handover is: an ID that was being reported last frame is gone from the
        screen this frame, while a NEW ID has appeared in the same place. The
        object never left; only its label did.
        """
        # Second failure shape: the ID did not vanish, it JUMPED to a different
        # object. Cross-assignment swaps two labels while both stay on screen,
        # which the vanish/appear test below would never notice.
        for tid, new_box in current_boxes.items():
            old_box = self._previous_reported.get(tid)
            if old_box is None:
                continue
            displacement = math.dist(center(old_box), center(new_box))
            if iou(old_box, new_box) == 0.0 and displacement > 1.5 * mean_side(old_box):
                self._log_identity_jump(record, tid, old_box, new_box, displacement)

        vanished = {
            tid: box
            for tid, box in self._previous_reported.items()
            if tid not in current_boxes
        }
        arrived = {
            tid: box for tid, box in current_boxes.items()
            if tid not in self._previous_reported
        }
        if not vanished or not arrived:
            return

        claimed: set[int] = set()
        for old_id, old_box in vanished.items():
            best_id, best_score = None, 0.0
            for new_id, new_box in arrived.items():
                if new_id in claimed:
                    continue
                overlap = iou(old_box, new_box)
                distance = math.dist(center(old_box), center(new_box))
                # Same place = overlapping, or centres within the object's own
                # scale. Both are generous; the log states which applied.
                if overlap > 0.05 or distance <= 2.0 * mean_side(old_box):
                    score = overlap + 1.0 / (1.0 + distance)
                    if score > best_score:
                        best_id, best_score = new_id, score
            if best_id is None:
                continue
            claimed.add(best_id)
            self._log_identity_handover(record, old_id, old_box, best_id)

    def _note(self, track: _ActiveTrack, event: str) -> None:
        """Append one lifecycle event to a track's history."""
        track.history.append(f"  frame {self._frame_number:<6d} {event}")
        if len(track.history) > 80:
            del track.history[: len(track.history) - 80]

    def _classify_failure(
        self,
        old_id: int,
        detections: list[Detection],
        matched_pairs: list[tuple[int, int]],
    ) -> tuple[str, list[str]]:
        """Determine WHY the old track stopped owning its bottle.

        Anchors on the track's LAST OBSERVED position -- where the bottle really
        was -- rather than its predicted position, so prediction error is
        measurable rather than assumed.

        Returns ``(cause, evidence_lines)``.
        """
        track = next((t for t in self._tracks if t.track_id == old_id), None)
        if track is None:
            history = self._graveyard.get(old_id)
            return (
                "TRACK EXPIRATION (lifecycle)",
                [
                    f"    track #{old_id} no longer exists - it was DELETED before "
                    "this frame after exceeding max_lost_frames="
                    f"{self._max_lost_frames}.",
                    f"    (its recorded history {'is below' if history else 'was lost'})",
                ],
            )

        if not detections:
            return (
                "NO DETECTIONS this frame",
                [f"    track #{old_id} could not match because YOLO returned nothing."],
            )

        anchor = track.last_observed_center or center(track.bbox)
        best = min(
            range(len(detections)),
            key=lambda j: math.dist(anchor, center(detections[j].bbox)),
        )
        detection = detections[best]
        predicted = track.bbox
        score = iou(predicted, detection.bbox)
        distance = math.dist(center(predicted), center(detection.bbox))
        gate = self._distance_gate(track, detection)
        object_moved = math.dist(anchor, center(detection.bbox))
        prediction_drift = math.dist(anchor, center(predicted))

        evidence = [
            f"    bottle's own detection this frame: det{best} {detection.bbox}",
            f"    track #{old_id} last OBSERVED at {track.last_observed_bbox} "
            f"(frame {track.last_observed_frame})",
            f"    track #{old_id} PREDICTED box this frame: {predicted}",
            f"    IoU(predicted, detection)      = {score:.3f}   "
            f"(threshold {self._iou_threshold:.2f})",
            f"    centre distance (predicted)    = {distance:.0f} px  "
            f"(gate {gate:.0f} px)",
            f"    bottle actually moved          = {object_moved:.0f} px "
            "since last observation",
            f"    prediction moved               = {prediction_drift:.0f} px "
            f"since last observation   [v=({track.vx:+.1f},{track.vy:+.1f})]",
            f"    age={track.frames_since_seen}  hits={track.hits}  "
            f"created frame {track.created_frame}",
        ]

        taken_by = {d: t for t, d in matched_pairs}
        if best in taken_by and self._tracks[taken_by[best]].track_id != old_id:
            thief = self._tracks[taken_by[best]].track_id
            return (
                "CROSS-ASSIGNMENT (another track took this bottle's detection)",
                evidence + [f"    det{best} was assigned to #{thief} instead."],
            )

        if score >= self._iou_threshold:
            return ("NOT A GATE FAILURE - IoU gate passed", evidence)
        if distance <= gate:
            return ("NOT A GATE FAILURE - distance gate passed", evidence)

        # Both gates failed. Was it the object moving, or the prediction drifting?
        if prediction_drift > 2.0 * object_moved + 20.0:
            return (
                "VELOCITY PREDICTION DRIFT",
                evidence
                + [
                    f"    the prediction travelled {prediction_drift:.0f}px while the "
                    f"bottle only moved {object_moved:.0f}px, so the predicted box "
                    "was looking in the wrong place.",
                ],
            )
        return (
            "DISTANCE GATE (object displacement exceeded the gate)",
            evidence
            + [
                f"    the bottle moved {object_moved:.0f}px, and even from the "
                f"predicted position the nearest detection was {distance:.0f}px "
                f"away, beyond the {gate:.0f}px gate.",
            ],
        )

    def _report_identity_switches(
        self,
        record: _FrameRecord,
        current_boxes: dict[int, BBox],
        detections: list[Detection],
        matched_pairs: list[tuple[int, int]],
    ) -> None:
        """Emit a full forensic report when a bottle changes ID.

        Recognises a switch even when the bottle went undetected for a while
        first: any ID newly on screen is checked against every identity shown in
        the recent past at that location.
        """
        newly_shown = [tid for tid in current_boxes if tid not in self._previous_reported]
        if not newly_shown:
            return

        for new_id in newly_shown:
            new_box = current_boxes[new_id]
            candidates = [
                (frame, box, tid)
                for tid, (frame, box) in self._identity_ledger.items()
                if tid != new_id
                and tid not in current_boxes
                and self._frame_number - frame <= _IDENTITY_LOOKBACK_FRAMES
                and math.dist(center(box), center(new_box)) <= 2.5 * mean_side(new_box)
            ]
            if not candidates:
                continue
            frame, old_box, old_id = max(candidates)
            self._log_switch_report(
                record, old_id, frame, old_box, new_id, new_box,
                detections, matched_pairs,
            )

    def _log_switch_report(
        self,
        record: _FrameRecord,
        old_id: int,
        old_frame: int,
        old_box: BBox,
        new_id: int,
        new_box: BBox,
        detections: list[Detection],
        matched_pairs: list[tuple[int, int]],
    ) -> None:
        """Print the complete forensic account of one identity switch."""
        cause, evidence = self._classify_failure(old_id, detections, matched_pairs)

        lines = [
            "",
            "#" * 78,
            f"IDENTITY SWITCH  #{old_id} -> #{new_id}   at frame {record.frame}",
            "#" * 78,
            f"  1. Frame number      : {record.frame}",
            f"  2. Old track ID      : #{old_id}   (last shown frame {old_frame} "
            f"at {old_box})",
            f"  3. New track ID      : #{new_id}  at {new_box}",
            f"  4. Detection now assigned to the old track: "
            f"{record.outcomes.get(old_id, 'track no longer exists')}",
            "",
            "  ROOT CAUSE: " + cause,
        ]
        lines.extend(evidence)
        lines += [
            "",
            "  13. Failure category checklist:",
        ]
        for label in (
            "IoU gate",
            "Distance gate",
            "Track expiration",
            "Track confirmation",
            "Velocity prediction drift",
            "Cross-assignment",
        ):
            hit = label.lower().split()[0] in cause.lower()
            lines.append(f"        [{'X' if hit else ' '}] {label}")
        lines.append(
            "        (track confirmation cannot be a cause here: this tracker "
            "shows every track from its first hit)"
        )
        lines += [
            "",
            f"  14. Why #{new_id} was created: its detection matched no existing "
            "track, so step 5 spawned a new id. Full per-track fates:",
        ]
        for tid, outcome in sorted(record.outcomes.items()):
            lines.append(f"        #{tid}: {outcome}")

        lines += ["", f"  LIFECYCLE TIMELINE - track #{old_id}:"]
        old_track = next((t for t in self._tracks if t.track_id == old_id), None)
        old_history = old_track.history if old_track else self._graveyard.get(old_id, [])
        lines.extend(old_history[-30:] or ["    (no history retained)"])

        lines += ["", f"  LIFECYCLE TIMELINE - track #{new_id}:"]
        new_track = next((t for t in self._tracks if t.track_id == new_id), None)
        lines.extend(
            (new_track.history[-10:] if new_track else []) or ["    (none)"]
        )

        lines += ["", "  ASSOCIATION AT THIS FRAME:"]
        lines.extend(f"    {row}" for row in (record.cost_matrix or ["(none)"]))
        lines.extend(record.assignment_report or [])
        lines.append("#" * 78)
        _log.info("\n".join(lines))

    def _log_roster(self, record: _FrameRecord, current_boxes: dict[int, BBox]) -> None:
        """Log which ID sits at which screen position, every frame.

        This is the primary evidence, and it needs no cleverness: reading the
        centre coordinates down the log shows an object holding one position
        while its ID changes underneath it. Close-together bottles make an
        erroneous re-assignment geometrically indistinguishable from real
        motion, so no automatic classifier can be trusted here -- but the raw
        position/ID timeline is unambiguous.

        Whenever the set of on-screen IDs changes, the full cost matrix is
        dumped too, because that is the frame where the decision was made.
        """
        roster = "  ".join(
            f"#{tid}@({int(center(box)[0])},{int(center(box)[1])})"
            for tid, box in sorted(current_boxes.items())
        )
        _log.info(
            "frame %-5d dets=%d  reported: %s",
            record.frame,
            len(record.detections),
            roster or "none",
        )

        if set(current_boxes) == set(self._previous_reported):
            return

        left = sorted(set(self._previous_reported) - set(current_boxes))
        joined = sorted(set(current_boxes) - set(self._previous_reported))
        if not self._previous_reported and not left:
            return  # cold start: the first objects appearing is not an event
        lines = [
            "",
            "-" * 78,
            f"ROSTER CHANGE at frame {record.frame}: "
            f"left screen {['#' + str(i) for i in left] or 'none'}, "
            f"joined {['#' + str(i) for i in joined] or 'none'}",
            "  Every track's fate this frame:",
        ]
        for tid, outcome in sorted(record.outcomes.items()):
            lines.append(f"    #{tid}: {outcome}")
        lines.append("  COST MATRIX (predicted boxes vs detections):")
        lines.extend(f"    {row}" for row in (record.cost_matrix or ["(none)"]))
        lines.append(
            "  Detections: " + ("  ".join(record.detections) or "none")
        )
        lines.append(
            "  Duplicate check: " + ("  ".join(record.duplicate_notes) or "none")
        )
        lines.append("")
        lines.extend(record.assignment_report or ["  (no assignment analysis)"])
        lines.append("-" * 78)
        _log.info("\n".join(lines))

    def _log_identity_jump(
        self,
        record: _FrameRecord,
        track_id: int,
        old_box: BBox,
        new_box: BBox,
        displacement: float,
    ) -> None:
        """Report a track whose box teleported -- it likely changed object.

        This is the signature of cross-assignment: the label stays on screen but
        moves onto a different bottle. Two of these in the same frame is an ID
        swap.
        """
        lines = [
            "",
            "=" * 78,
            f"IDENTITY JUMP at frame {record.frame}: "
            f"#{track_id} teleported {displacement:.0f}px with ZERO overlap",
            "=" * 78,
            f"  #{track_id} was at {old_box}, is now at {new_box}.",
            f"  A jump this large with no overlap means #{track_id} probably "
            "attached to a DIFFERENT object (cross-assignment).",
            f"  outcome: {record.outcomes.get(track_id, 'unknown')}",
            "",
            "  ASSOCIATION COST MATRIX (predicted boxes vs this frame's detections):",
        ]
        lines.extend(f"    {row}" for row in (record.cost_matrix or ["(none)"]))
        lines.append(
            "  Detections this frame: " + ("  ".join(record.detections) or "none")
        )
        lines.append(
            "  Duplicate check: " + ("  ".join(record.duplicate_notes) or "none")
        )
        lines.append("=" * 78)
        _log.info("\n".join(lines))

    def _log_identity_handover(
        self, record: _FrameRecord, old_id: int, old_box: BBox, new_id: int
    ) -> None:
        """Print the full forensic account of one ID handover."""
        old_outcome = record.outcomes.get(old_id, "track no longer exists")
        predicted = record.predicted_boxes.get(old_id)

        lines = [
            "",
            "=" * 78,
            f"IDENTITY HANDOVER at frame {record.frame}: "
            f"#{old_id} -> #{new_id}  (object stayed visible)",
            "=" * 78,
            f"  Object was reported as #{old_id} at {old_box} on the previous frame.",
            f"  It is now reported as #{new_id}.",
            "",
            f"  WHY #{old_id} WAS NOT MATCHED:",
            f"    predicted box this frame: {predicted if predicted else 'n/a'}",
            f"    outcome: {old_outcome}",
            "",
            f"  WHY #{new_id} WAS CREATED:",
            "    its detection matched no existing track, so step 5 spawned a new id.",
            "",
            "  ASSOCIATION COST MATRIX (predicted boxes vs this frame's detections):",
        ]
        lines.extend(f"    {row}" for row in (record.cost_matrix or ["(none)"]))
        lines.append("")
        lines.append(
            f"  Detections this frame: "
            + ("  ".join(record.detections) or "none")
        )
        lines.append(
            "  Duplicate check: "
            + ("  ".join(record.duplicate_notes) or "none")
        )
        lines.append("")
        lines.append("  Preceding frames:")
        for past in list(self._history)[:-1]:
            lines.append(
                f"    frame {past.frame}: dets={len(past.detections)} "
                f"matches=[{'  '.join(past.matches) or 'none'}] "
                f"reported={past.reported}"
            )
        lines.append("=" * 78)
        _log.info("\n".join(lines))

    def _predict(self) -> None:
        """Shift each track's box by its estimated velocity (step 0).

        Tracks with only a single observation have no velocity yet, so they are
        left where they are.

        An unobserved track keeps coasting at its FULL last-known velocity. This
        is deliberate and load-bearing: a bottle that the detector loses (motion
        blur, or a hand covering it) is still physically moving, so the estimate
        must move with it. An earlier version damped the velocity while blind,
        which made the predicted box fall progressively behind the real bottle
        and caused re-acquisition to fail -- a new ID after every dropout.
        Coasting is bounded by ``max_lost_frames``, and a coasting track cannot
        steal an observed bottle because that bottle's own track scores a far
        higher IoU and greedy matching consumes it first.
        """
        for track in self._tracks:
            if track.hits < 2:
                continue

            dx = int(round(track.vx))
            dy = int(round(track.vy))
            if dx or dy:
                x1, y1, x2, y2 = track.bbox
                track.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _absorb(self, track: _ActiveTrack, detection: Detection) -> None:
        """Update a matched track from its new observation (step 3).

        Velocity is measured between the last *observed* centre and this one,
        divided by however many frames elapsed, then blended into the running
        estimate so a single noisy detection cannot throw the track off.
        """
        new_center = center(detection.bbox)

        if track.last_observed_center is not None:
            steps = track.frames_since_seen + 1
            observed_vx = (new_center[0] - track.last_observed_center[0]) / steps
            observed_vy = (new_center[1] - track.last_observed_center[1]) / steps
            if track.hits < 2:
                # First measurement: adopt it outright, there is nothing to blend.
                track.vx, track.vy = observed_vx, observed_vy
            else:
                alpha = self._velocity_smoothing
                track.vx = alpha * observed_vx + (1.0 - alpha) * track.vx
                track.vy = alpha * observed_vy + (1.0 - alpha) * track.vy

        track.last_observed_center = new_center
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.class_name = detection.class_name
        track.frames_since_seen = 0
        track.hits += 1

    def _distance_gate(self, track: _ActiveTrack, detection: Detection) -> float:
        """How far this track's centre may be from a detection and still match.

        The gate scales with the object's size -- a big bottle can travel
        further between frames than a small one and still be the same bottle --
        and is widened for a track that has no velocity estimate yet, because
        prediction is doing none of the work for it.
        """
        gate = self._max_center_distance_factor * mean_side(detection.bbox)
        if track.hits < 2:
            gate *= _BOOTSTRAP_GATE_MULTIPLIER
        return gate

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], set[int], dict[tuple[int, int], str]]:
        """Match tracks to detections in two stages.

        Stage 1 uses IoU against the predicted boxes -- precise, and the right
        signal whenever the boxes still overlap. Stage 2 catches whatever is
        left using centre distance, gated by the object's own size, because a
        fast-moving bottle can produce boxes that do not overlap at all.

        Args:
            detections: This frame's detections.

        Returns:
            ``(pairs, matched_detection_indices, stage_labels)``.
        """
        pairs: list[tuple[int, int]] = []
        claimed_tracks: set[int] = set()
        claimed_detections: set[int] = set()
        stages: dict[tuple[int, int], str] = {}

        # --- Stage 1: IoU on predicted boxes, strongest overlap first. ---
        iou_candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                score = iou(track.bbox, detection.bbox)
                if score >= self._iou_threshold:
                    iou_candidates.append((score, track_index, detection_index))
        iou_candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        for score, track_index, detection_index in iou_candidates:
            if track_index in claimed_tracks or detection_index in claimed_detections:
                continue
            claimed_tracks.add(track_index)
            claimed_detections.add(detection_index)
            pairs.append((track_index, detection_index))
            stages[(track_index, detection_index)] = f"IoU {score:.2f}"

        # --- Stage 2: centre distance for whatever stage 1 missed, nearest first. ---
        distance_candidates: list[tuple[float, int, int, float]] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in claimed_tracks:
                continue
            track_center = center(track.bbox)
            for detection_index, detection in enumerate(detections):
                if detection_index in claimed_detections:
                    continue
                detection_center = center(detection.bbox)
                distance = math.dist(track_center, detection_center)
                gate = self._distance_gate(track, detection)
                if distance <= gate:
                    distance_candidates.append(
                        (distance, track_index, detection_index, gate)
                    )
        distance_candidates.sort(key=lambda candidate: candidate[0])

        for distance, track_index, detection_index, gate in distance_candidates:
            if track_index in claimed_tracks or detection_index in claimed_detections:
                continue
            claimed_tracks.add(track_index)
            claimed_detections.add(detection_index)
            pairs.append((track_index, detection_index))
            stages[(track_index, detection_index)] = (
                f"dist {distance:.0f}<={gate:.0f}"
            )

        return pairs, claimed_detections, stages

    # ------------------------------------------------------------------
    # Diagnostics (DEBUG level only; no effect on the algorithm)
    # ------------------------------------------------------------------

    def _log_spawn_reason(
        self,
        new_id: int,
        detection: Detection,
        detection_index: int,
        tracks: list[_ActiveTrack],
        claimed_by: dict[int, int],
    ) -> None:
        """Explain why a detection had to become a NEW id instead of matching.

        Fires only when an ID actually changes, so it stays quiet in normal
        operation. For every existing track it states an explicit verdict:
        whether the track was already taken by another detection, or which gate
        it failed and by how much. Comparison operators reflect the ACTUAL
        relation, so a satisfied gate can never read as a failure.
        """
        detection_center = center(detection.bbox)

        ranked = sorted(
            range(len(tracks)),
            key=lambda i: math.dist(center(tracks[i].bbox), detection_center),
        )[:_SPAWN_DIAGNOSTIC_TRACKS]

        lines = []
        for index in ranked:
            track = tracks[index]
            score = iou(track.bbox, detection.bbox)
            distance = math.dist(center(track.bbox), detection_center)
            gate = self._distance_gate(track, detection)

            if index in claimed_by:
                verdict = f"UNAVAILABLE - already matched det{claimed_by[index]}"
            elif score >= self._iou_threshold:
                verdict = "IoU gate PASSED but was not used (unexpected)"
            elif distance <= gate:
                verdict = "distance gate PASSED but was not used (unexpected)"
            else:
                verdict = (
                    f"rejected - iou {score:.2f} < {self._iou_threshold:.2f} "
                    f"AND dist {distance:.0f} > gate {gate:.0f}"
                )

            lines.append(
                f"#{track.track_id} box={track.bbox} v=({track.vx:+.1f},{track.vy:+.1f}) "
                f"age={track.frames_since_seen} hits={track.hits} "
                f"iou={score:.2f} dist={distance:.0f} gate={gate:.0f} -> {verdict}"
            )

        # DEBUG, not INFO: a new id is often legitimate (a bottle entering the
        # scene). The event worth shouting about is an IDENTITY HANDOVER.
        _log.debug(
            "NEW ID #%d for det%d %s (frame %d, %d tracks existed). Candidates:\n    %s",
            new_id,
            detection_index,
            detection.bbox,
            self._frame_number,
            len(tracks),
            "\n    ".join(lines),
        )

    def _pair_cost(
        self, track: _ActiveTrack, detection: Detection
    ) -> tuple[bool, float, str]:
        """Return ``(feasible, cost, why)`` for one track/detection pair.

        The cost mirrors the tracker's own two-stage preference on a single
        scale so greedy and optimal assignments are directly comparable.
        """
        score = iou(track.bbox, detection.bbox)
        distance = math.dist(center(track.bbox), center(detection.bbox))
        gate = self._distance_gate(track, detection)

        if score >= self._iou_threshold:
            return True, 1.0 - score, f"iou={score:.3f}"
        if distance <= gate:
            return True, 1.0 + distance / max(gate, 1.0), f"d={distance:.0f}/{gate:.0f}"
        return False, math.inf, f"blocked(iou={score:.2f} d={distance:.0f}>{gate:.0f})"

    def _optimal_assignment(
        self, costs: list[list[float]], size: int
    ) -> tuple[list[tuple[int, int]], float, float]:
        """Exact minimum-cost assignment by permutation search.

        ``costs`` is padded to ``size x size`` with the unmatched penalty, so the
        result is a perfect matching whose real pairs are the cells costing less
        than the penalty.

        Returns ``(best_pairs, best_total, runner_up_total)`` where the runner-up
        is the cheapest assignment producing a DIFFERENT set of real pairs. When
        the runner-up ties the best, the frame is genuinely ambiguous and no
        assignment algorithm -- greedy, Hungarian or otherwise -- can be relied
        on to choose correctly.
        """
        totals_by_pairing: dict[frozenset[tuple[int, int]], float] = {}
        for perm in itertools.permutations(range(size)):
            total = 0.0
            for row, col in enumerate(perm):
                total += costs[row][col]
            real = frozenset(
                (row, col)
                for row, col in enumerate(perm)
                if costs[row][col] < _ASSIGNMENT_PENALTY
            )
            if total < totals_by_pairing.get(real, math.inf):
                totals_by_pairing[real] = total

        ranked = sorted(totals_by_pairing.items(), key=lambda item: item[1])
        best_set, best_total = ranked[0]
        runner_total = ranked[1][1] if len(ranked) > 1 else math.inf
        return sorted(best_set), best_total, runner_total

    def _analyse_assignment(
        self,
        record: _FrameRecord,
        detections: list[Detection],
        matched_pairs: list[tuple[int, int]],
    ) -> None:
        """Compare the tracker's greedy assignment against the global optimum.

        Diagnostic only. Answers directly: on THIS frame, did greedy matching
        pick a worse set of pairs than an optimal solver would have?
        """
        n_tracks, n_detections = len(self._tracks), len(detections)
        if not n_tracks or not n_detections:
            return

        size = max(n_tracks, n_detections)
        if size > _MAX_EXACT_ASSIGNMENT:
            record.assignment_report = [
                f"assignment analysis skipped: {size} > {_MAX_EXACT_ASSIGNMENT}"
            ]
            return

        # Padded square cost matrix; padding and infeasible cells cost the
        # unmatched penalty.
        costs = [[_ASSIGNMENT_PENALTY] * size for _ in range(size)]
        why = [["-"] * size for _ in range(size)]
        for i in range(n_tracks):
            for j in range(n_detections):
                feasible, cost, reason = self._pair_cost(self._tracks[i], detections[j])
                why[i][j] = reason
                if feasible:
                    costs[i][j] = cost

        greedy_total = sum(costs[i][j] for i, j in matched_pairs) + (
            _ASSIGNMENT_PENALTY * (size - len(matched_pairs))
        )
        optimal_pairs, optimal_total, runner_total = self._optimal_assignment(
            costs, size
        )
        optimal_pairs = [
            (i, j) for i, j in optimal_pairs if i < n_tracks and j < n_detections
        ]

        def render(pairs: list[tuple[int, int]]) -> str:
            return (
                "  ".join(
                    f"#{self._tracks[i].track_id}<-det{j}({costs[i][j]:.3f})"
                    for i, j in sorted(pairs)
                )
                or "nothing matched"
            )

        gap = greedy_total - optimal_total
        record.greedy_suboptimal = gap > 1e-9

        lines = [
            "  UNIFIED COST MATRIX  (stage1 = 1-iou in [0,0.8];  "
            "stage2 = 1+d/gate in [1,2];  unmatched/blocked = 3.000)",
            "    " + "track".ljust(30) + "".join(
                f"det{j}".ljust(26) for j in range(n_detections)
            ),
        ]
        for i in range(n_tracks):
            row = f"#{self._tracks[i].track_id} age={self._tracks[i].frames_since_seen}".ljust(30)
            for j in range(n_detections):
                cell = costs[i][j]
                shown = f"{cell:.3f}" if cell < _ASSIGNMENT_PENALTY else "BLOCKED"
                row += f"{shown} [{why[i][j]}]".ljust(26)
            lines.append("    " + row)

        margin = runner_total - optimal_total
        ambiguous = margin <= 1e-9

        lines += [
            "",
            f"  GREEDY  chose: {render(matched_pairs)}",
            f"          total cost = {greedy_total:.3f}",
            f"  OPTIMAL would choose: {render(optimal_pairs)}",
            f"          total cost = {optimal_total:.3f}",
            f"  Runner-up assignment total = "
            + ("n/a" if math.isinf(runner_total) else f"{runner_total:.3f}")
            + f"   (margin over optimal = "
            + ("n/a" if math.isinf(runner_total) else f"{margin:.3f})"),
        ]
        if record.greedy_suboptimal:
            lines.append(
                f"  VERDICT: GREEDY IS SUBOPTIMAL on this frame "
                f"(worse by {gap:.3f}). An optimal solver WOULD have chosen "
                "differently and better."
            )
        elif ambiguous:
            lines.append(
                "  VERDICT: AMBIGUOUS - a different assignment scores EXACTLY the "
                "same. Greedy is not at fault here, and an optimal solver would "
                "have no basis to choose correctly either."
            )
        else:
            lines.append(
                f"  VERDICT: greedy already matched the optimal assignment "
                f"(next-best is {margin:.3f} worse, so the choice was clear-cut)."
            )
        record.assignment_report = lines

    def _record_association(
        self,
        record: _FrameRecord,
        detections: list[Detection],
        matched_pairs: list[tuple[int, int]],
        stages: dict[tuple[int, int], str],
    ) -> None:
        """Capture the full cost matrix and every track's fate for this frame.

        Must be called after :meth:`_associate` but BEFORE ``_absorb``, while
        the tracks still hold the predicted boxes that association actually
        used.
        """
        pair_for_track = {t_idx: d_idx for t_idx, d_idx in matched_pairs}

        header = "track".ljust(34) + "".join(
            f"det{i}".ljust(30) for i in range(len(detections))
        )
        rows = [header] if detections else []

        for track_index, track in enumerate(self._tracks):
            record.predicted_boxes[track.track_id] = track.bbox
            label = (
                f"#{track.track_id} age={track.frames_since_seen} "
                f"hits={track.hits} v=({track.vx:+.0f},{track.vy:+.0f})"
            ).ljust(34)

            cells = []
            for detection_index, detection in enumerate(detections):
                score = iou(track.bbox, detection.bbox)
                distance = math.dist(center(track.bbox), center(detection.bbox))
                gate = self._distance_gate(track, detection)
                if score >= self._iou_threshold:
                    verdict = "IoU-ok"
                elif distance <= gate:
                    verdict = "dist-ok"
                else:
                    verdict = "BLOCKED"
                cells.append(
                    f"iou={score:.2f} d={distance:.0f}/{gate:.0f} {verdict}".ljust(30)
                )
            rows.append(label + "".join(cells))

            # This track's fate, stated plainly.
            if track_index in pair_for_track:
                detection_index = pair_for_track[track_index]
                record.outcomes[track.track_id] = (
                    f"MATCHED det{detection_index} via "
                    f"{stages[(track_index, detection_index)]}"
                )
            elif detections:
                best = min(
                    range(len(detections)),
                    key=lambda i: math.dist(
                        center(track.bbox), center(detections[i].bbox)
                    ),
                )
                score = iou(track.bbox, detections[best].bbox)
                distance = math.dist(
                    center(track.bbox), center(detections[best].bbox)
                )
                gate = self._distance_gate(track, detections[best])
                taken_by = {d: t for t, d in matched_pairs}
                if best in taken_by:
                    thief = self._tracks[taken_by[best]].track_id
                    record.outcomes[track.track_id] = (
                        f"NOT MATCHED - nearest det{best} was taken by #{thief} "
                        f"(CROSS-ASSIGNMENT); own scores iou={score:.2f} "
                        f"dist={distance:.0f} gate={gate:.0f}"
                    )
                else:
                    record.outcomes[track.track_id] = (
                        f"NOT MATCHED - nearest det{best} failed both gates: "
                        f"iou={score:.2f} < {self._iou_threshold:.2f} and "
                        f"dist={distance:.0f} > gate={gate:.0f}"
                    )
            else:
                record.outcomes[track.track_id] = "NOT MATCHED - no detections"

        record.cost_matrix = rows

    @staticmethod
    def _duplicate_notes(detections: list[Detection]) -> list[str]:
        """Flag detections in THIS frame that appear to cover one object twice.

        The tracker sees detections after class filtering, so this answers the
        question that actually matters for identity: did the tracker receive two
        boxes for a single bottle? A second box has nowhere to go except into a
        newly spawned track.
        """
        notes: list[str] = []
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                overlap = iou(detections[i].bbox, detections[j].bbox)
                inside = containment(detections[i].bbox, detections[j].bbox)
                if overlap >= 0.30 or inside >= 0.60:
                    notes.append(
                        f"det{i}+det{j} iou={overlap:.2f} containment={inside:.2f}"
                        "  <-- SAME OBJECT DETECTED TWICE"
                    )
        return notes

    def _render_tracks(self, as_list: bool = False) -> list[str] | str:
        """Compact rendering of the current tracks."""
        rendered = [
            f"#{t.track_id}{t.bbox}v=({t.vx:+.1f},{t.vy:+.1f})"
            f"age={t.frames_since_seen}hits={t.hits}"
            for t in self._tracks
        ]
        if as_list:
            return rendered
        return rendered or "none"

    def _log_frame_header(self, detections: list[Detection]) -> None:
        """Emit the per-frame diagnostic block (DEBUG only)."""
        now = time.perf_counter()
        if self._last_update_time is None:
            elapsed_ms = 0.0
            fps = 0.0
        else:
            elapsed_ms = (now - self._last_update_time) * 1000.0
            fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        self._last_update_time = now

        _log.debug(
            "--- frame %d | %.0f ms since last frame (%.1f FPS) ---",
            self._frame_number,
            elapsed_ms,
            fps,
        )
        _log.debug("  tracks PREDICTED (%d): %s", len(self._tracks), self._render_tracks())
        _log.debug(
            "  detections IN (%d): %s",
            len(detections),
            [f"det{i}{d.bbox}" for i, d in enumerate(detections)] or "none",
        )

        if not self._tracks or not detections:
            _log.debug("  IoU matrix: (nothing to compare)")
            return
        _log.debug(
            "  IoU matrix (stage-1 threshold=%.2f):", self._iou_threshold
        )
        for track in self._tracks:
            track_center = center(track.bbox)
            cells = []
            for i, detection in enumerate(detections):
                score = iou(track.bbox, detection.bbox)
                distance = math.dist(track_center, center(detection.bbox))
                gate = self._max_center_distance_factor * mean_side(detection.bbox)
                flag = ""
                if score >= self._iou_threshold:
                    flag = " <-IoU"
                elif distance <= gate:
                    flag = " <-DIST"
                cells.append(f"det{i}: iou={score:.3f} d={distance:.0f}/{gate:.0f}{flag}")
            _log.debug("    track #%-3d %s -> %s", track.track_id, track.bbox, " | ".join(cells))
