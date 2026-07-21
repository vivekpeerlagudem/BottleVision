"""Diagnostic observability for the pipeline.

Responsibility: record what each stage did to a frame and report it as one
coherent trace, plus save the raw frames behind a dropout so the question "was
the bottle visible, occluded, or out of shot?" is answered by looking at the
image rather than by inference.

Everything here is strictly observational. No stage changes its behaviour when
these objects are present; they only receive counts that were computed anyway.

The frame number used throughout is the PIPELINE's, so every diagnostic line
across every stage refers to the same frame. Stages deliberately do not keep
their own counters, which could drift apart.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bottlevision.detection import Detection
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class DetectorTrace:
    """What YOLO returned for one frame, and what survived the threshold."""

    query_confidence: float
    threshold: float
    raw: list[Detection] = field(default_factory=list)
    kept: list[Detection] = field(default_factory=list)

    @property
    def dropped_by_confidence(self) -> list[Detection]:
        """Boxes YOLO found that the configured threshold discarded."""
        kept_ids = {id(d) for d in self.kept}
        return [d for d in self.raw if id(d) not in kept_ids]


@dataclass
class FilterTrace:
    """What the class filter received and what it discarded."""

    target_class: str
    received: int = 0
    kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)


class FrameDumper:
    """Saves selected frames as JPEGs for later inspection."""

    def __init__(self, directory: Path, limit: int = 400) -> None:
        """Prepare an output directory.

        Args:
            directory: Where frames are written.
            limit: Stop after this many frames so a long run cannot fill the disk.
        """
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._limit = limit
        self._saved = 0
        _log.info("Dropout frames will be saved to %s", self._directory.resolve())

    def save(self, frame: np.ndarray, frame_number: int, tag: str) -> Path | None:
        """Write one frame exactly as captured, named so it sorts by frame number.

        Args:
            frame: BGR image to write, unmodified.
            frame_number: Pipeline frame number, used in the filename.
            tag: Short reason, included in the filename.

        Returns:
            The path written, or ``None`` if nothing was written.
        """
        if self._saved >= self._limit:
            return None
        path = self._directory / f"frame_{frame_number:06d}_{tag}.jpg"
        if not cv2.imwrite(str(path), frame):
            _log.warning("Could not write diagnostic frame to %s", path)
            return None
        self._saved += 1
        if self._saved == self._limit:
            _log.warning(
                "Frame dump limit (%d) reached; no more frames will be saved.",
                self._limit,
            )
        return path


class PipelineTracer:
    """Reports a frame's journey through every stage of the pipeline."""

    def __init__(self, dump_directory: Path | None = None) -> None:
        """Create a tracer, optionally saving frames that yield no bottle.

        Args:
            dump_directory: Where to write frames for which the tracker received
                zero detections. ``None`` disables saving.
        """
        self._dumper = FrameDumper(dump_directory) if dump_directory else None

    def trace(
        self,
        frame_number: int,
        frame: np.ndarray,
        detector_trace: DetectorTrace | None,
        filter_trace: FilterTrace | None,
        tracker_received: int,
    ) -> None:
        """Log one frame's stage-by-stage counts.

        Frames that reach the tracker with detections get a single compact line.
        Frames that reach it with NOTHING -- the dropouts we are chasing -- get
        the full trace, a stage-by-stage breakdown, a conclusion naming where the
        bottle was lost, and the raw frame written to disk.
        """
        height, width = frame.shape[:2]
        raw_count = len(detector_trace.raw) if detector_trace else -1
        confidence_kept = len(detector_trace.kept) if detector_trace else -1
        filter_kept = filter_trace.kept if filter_trace else -1
        query = detector_trace.query_confidence if detector_trace else 0.0
        threshold = detector_trace.threshold if detector_trace else 0.0

        if tracker_received > 0:
            _log.info(
                "Frame %-6d Camera: 1 frame (%dx%d) | YOLO@%.2f: %d | "
                "Confidence kept (>=%.2f): %d | Bottle filter kept: %d | "
                "Tracker received: %d bottle(s)",
                frame_number, width, height, query, raw_count,
                threshold, confidence_kept, filter_kept, tracker_received,
            )
            return

        lines = [
            "",
            "-" * 70,
            f"Frame {frame_number}   *** TRACKER RECEIVED ZERO BOTTLES ***",
            f"  Camera:             1 frame ({width}x{height})",
            f"  YOLO@{query:.2f}:          {raw_count} detections",
            f"  Confidence kept:    {confidence_kept}   (threshold {threshold:.2f})",
            f"  Bottle filter kept: {filter_kept}"
            + (f"   (target '{filter_trace.target_class}')" if filter_trace else ""),
            f"  Tracker received:   {tracker_received} bottles",
        ]

        if detector_trace:
            dropped = detector_trace.dropped_by_confidence
            lines.append(
                "  Dropped by confidence: "
                + (
                    "  ".join(
                        f"{d.class_name} {d.confidence:.2f} {d.bbox}" for d in dropped
                    )
                    or "none"
                )
            )
        if filter_trace:
            lines.append(
                "  Dropped by class filter: "
                + (
                    "  ".join(f"{name} x{n}" for name, n in filter_trace.dropped.items())
                    or "none"
                )
            )

        lines.append("  CONCLUSION: " + self._conclude(detector_trace, filter_trace))

        if self._dumper is not None:
            path = self._dumper.save(frame, frame_number, "nobottle")
            if path is not None:
                lines.append(f"  Raw frame saved -> {path}")
        lines.append("-" * 70)
        _log.info("\n".join(lines))

    @staticmethod
    def _conclude(
        detector_trace: DetectorTrace | None, filter_trace: FilterTrace | None
    ) -> str:
        """Name the stage at which the bottle was lost."""
        if detector_trace is None:
            return "no detector trace available."
        if not detector_trace.raw:
            return (
                f"YOLO DETECTOR DROPOUT - nothing found at all, even at "
                f"conf={detector_trace.query_confidence:.2f}. Inspect the saved "
                "frame for occlusion, blur, or the bottle being out of shot."
            )
        if not detector_trace.kept:
            best = max(detector_trace.raw, key=lambda d: d.confidence)
            return (
                f"LOST AT CONFIDENCE FILTER - YOLO found {len(detector_trace.raw)} "
                f"box(es), best was '{best.class_name}' at {best.confidence:.2f}, "
                f"below the {detector_trace.threshold:.2f} threshold."
            )
        if filter_trace is not None and filter_trace.kept == 0:
            seen = "  ".join(
                f"{name} x{n}" for name, n in filter_trace.dropped.items()
            )
            return (
                f"LOST AT CLASS FILTER - {filter_trace.received} detection(s) "
                f"passed confidence but none were '{filter_trace.target_class}'. "
                f"Classes seen: {seen or 'none'}."
            )
        return (
            "PIPELINE BUG - detections survived every filter yet the tracker "
            "received none. This should be impossible."
        )
