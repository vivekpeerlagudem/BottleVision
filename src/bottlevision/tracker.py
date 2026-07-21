"""Multi-object tracking.

Responsibility: give detections a persistent identity across frames. The
detector and post-processor are memoryless -- they describe a single instant.
This module holds the only state that spans time.

Algorithm: constant-velocity prediction with two-stage association.
Each frame:

    0. PREDICT -- shift every track's box by its estimated velocity, so we
                  match against where the object *is now*, not where it *was*.
    1. STAGE-1 -- IoU between each predicted box and each detection.
    2. STAGE-2 -- fall back to centre distance for whatever stage 1 missed.
                  IoU collapses to exactly 0.0 once boxes stop overlapping, so
                  no IoU threshold can rescue fast motion; centre distance
                  degrades gracefully and can.
    3. UPDATE  -- matched tracks adopt the observed box and refine velocity.
    4. AGE     -- unmatched tracks get older by one frame.
    5. SPAWN   -- unmatched detections become new tracks with fresh IDs.
    6. RETIRE  -- tracks unseen longer than ``max_lost_frames`` are dropped.
    7. REPORT  -- only tracks confirmed in this frame are returned.

A lost track keeps coasting on its last velocity, so a briefly occluded object
is re-acquired at its predicted position. Lost tracks are not reported --
their position is an estimate, so drawing them would be a guess.

Known limits (motion-only tracker): if two objects physically exchange places
while occluding each other, geometry cannot tell them apart and IDs may swap.
Solving that requires appearance-based re-identification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bottlevision.config import TrackerConfig
from bottlevision.detection import Detection, Track
from bottlevision.geometry import BBox, center, iou, mean_side
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)

# A track that has been observed only once has no velocity yet, so prediction
# cannot help it and its distance gate must absorb the object's ENTIRE
# displacement between frames. Established tracks get the tight gate because
# prediction has already moved them most of the way. Without this widening a
# fast-moving object can never reach its second observation, so it never earns
# a velocity, so it never predicts -- a deadlock that spawns a fresh ID every
# frame.
_BOOTSTRAP_GATE_MULTIPLIER = 3.0


@dataclass
class _ActiveTrack:
    """Internal, mutable bookkeeping for one tracked object.

    Attributes:
        track_id: Stable integer identity.
        bbox: Current position estimate -- the observed box after a match, or
            the velocity-predicted box while unmatched.
        confidence: Confidence of the most recent observation.
        class_name: Human-readable class label.
        frames_since_seen: 0 the frame after a match, then +1 per missed frame.
        vx, vy: Estimated centre velocity in pixels per frame.
        hits: How many times this track has been observed. Velocity is only
            trusted (and prediction only applied) from the second hit onward.
        last_observed_center: Centre of the last observed box, used to measure
            velocity across however many frames have elapsed since.
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


class Tracker:
    """Assigns persistent integer IDs to detections across frames."""

    def __init__(self, config: TrackerConfig) -> None:
        """Start with no tracks and the first ID equal to 1."""
        self._iou_threshold = config.iou_threshold
        self._max_center_distance_factor = config.max_center_distance_factor
        self._velocity_smoothing = config.velocity_smoothing
        self._max_lost_frames = config.max_lost_frames
        self._tracks: list[_ActiveTrack] = []
        self._next_id = 1
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
            detections: This frame's detections (already class-filtered).

        Returns:
            The tracks confirmed in this frame, each with a stable ``track_id``.
            Tracks that are currently lost are omitted.
        """
        self._predict()

        matched_pairs, matched_detections = self._associate(detections)
        matched_tracks = {track_index for track_index, _ in matched_pairs}

        # 3. Matched tracks adopt their new observation and refine velocity.
        for track_index, detection_index in matched_pairs:
            self._absorb(self._tracks[track_index], detections[detection_index])

        # 4. Tracks with no match this frame grow older.
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.frames_since_seen += 1

        # 5. Detections with no match are new objects.
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            self._tracks.append(
                _ActiveTrack(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    last_observed_center=center(detection.bbox),
                )
            )
            self._next_id += 1

        # 6. Retire tracks that have been unseen for too long.
        self._tracks = [
            track
            for track in self._tracks
            if track.frames_since_seen <= self._max_lost_frames
        ]

        # 7. Report only what was actually seen in this frame.
        return [
            Track(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                class_name=track.class_name,
            )
            for track in self._tracks
            if track.frames_since_seen == 0
        ]

    def _predict(self) -> None:
        """Shift each track's box by its estimated velocity (step 0).

        Tracks with only a single observation have no velocity yet, so they are
        left where they are. Unobserved tracks keep coasting at their full
        last-known velocity -- a briefly occluded object is still physically
        moving, so the estimate must move with it. Coasting is bounded by
        ``max_lost_frames``, and a coasting track cannot steal an observed
        object because that object's own track scores a far higher IoU and
        greedy matching consumes it first.
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

        Velocity is measured between the last observed centre and this one,
        divided by however many frames elapsed, then blended into the running
        estimate so a single noisy detection cannot throw the track off.
        """
        new_center = center(detection.bbox)

        if track.last_observed_center is not None:
            steps = track.frames_since_seen + 1
            observed_vx = (new_center[0] - track.last_observed_center[0]) / steps
            observed_vy = (new_center[1] - track.last_observed_center[1]) / steps
            if track.hits < 2:
                # First real measurement: adopt it outright.
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

        The gate scales with object size -- a big object may travel further
        between frames than a small one -- and is widened for a track that has
        no velocity estimate yet, because prediction is doing none of the work
        for it.
        """
        gate = self._max_center_distance_factor * mean_side(detection.bbox)
        if track.hits < 2:
            gate *= _BOOTSTRAP_GATE_MULTIPLIER
        return gate

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], set[int]]:
        """Match tracks to detections in two stages.

        Stage 1 uses IoU against the predicted boxes -- precise, and the right
        signal whenever the boxes still overlap. Stage 2 catches whatever is
        left using centre distance, gated by object size, because a fast-moving
        object can produce boxes that do not overlap at all.

        Returns:
            ``(pairs, matched_detection_indices)`` where ``pairs`` holds
            ``(track_index, detection_index)`` matches.
        """
        pairs: list[tuple[int, int]] = []
        claimed_tracks: set[int] = set()
        claimed_detections: set[int] = set()

        # Stage 1: IoU on predicted boxes, strongest overlap first.
        iou_candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                score = iou(track.bbox, detection.bbox)
                if score >= self._iou_threshold:
                    iou_candidates.append((score, track_index, detection_index))
        iou_candidates.sort(key=lambda c: c[0], reverse=True)
        for _score, track_index, detection_index in iou_candidates:
            if track_index in claimed_tracks or detection_index in claimed_detections:
                continue
            claimed_tracks.add(track_index)
            claimed_detections.add(detection_index)
            pairs.append((track_index, detection_index))

        # Stage 2: centre distance for whatever stage 1 missed, nearest first.
        distance_candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in claimed_tracks:
                continue
            track_center = center(track.bbox)
            for detection_index, detection in enumerate(detections):
                if detection_index in claimed_detections:
                    continue
                distance = math.dist(track_center, center(detection.bbox))
                if distance <= self._distance_gate(track, detection):
                    distance_candidates.append((distance, track_index, detection_index))
        distance_candidates.sort(key=lambda c: c[0])
        for _distance, track_index, detection_index in distance_candidates:
            if track_index in claimed_tracks or detection_index in claimed_detections:
                continue
            claimed_tracks.add(track_index)
            claimed_detections.add(detection_index)
            pairs.append((track_index, detection_index))

        return pairs, claimed_detections
