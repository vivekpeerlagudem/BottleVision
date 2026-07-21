"""Object detector.

Responsibility: load the pre-trained YOLO model ONCE and run it on a single
frame, returning raw detections (bounding box, class, confidence) for EVERY
object it finds. This module intentionally knows nothing about "bottles"
specifically -- filtering happens later, in post-processing.

The model is loaded in ``__init__`` and reused for every call to :meth:`detect`,
which is exactly what the real-time webcam loop will need.

DIAGNOSTICS (M5 investigation): this module reports what YOLO actually emitted
each frame, and flags pairs of detections that appear to cover the SAME physical
object. Two boxes for one bottle are invisible on screen (they overlap) but are
not invisible to the tracker, which sees an extra unmatched detection. Duplicate
alerts are logged at INFO so they appear without ``--debug``; the full per-frame
listing is DEBUG.
"""

from __future__ import annotations

import logging

import numpy as np
from ultralytics import YOLO

from bottlevision.config import DetectorConfig
from bottlevision.detection import Detection
from bottlevision.diagnostics import DetectorTrace
from bottlevision.geometry import containment, iou
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)

# Two same-class boxes are treated as one physical object seen twice when they
# overlap substantially, OR when one is largely swallowed by the other (which
# IoU alone under-reports).
_DUPLICATE_IOU = 0.30
_DUPLICATE_CONTAINMENT = 0.60


class Detector:
    """Wraps an Ultralytics YOLO model and returns framework-agnostic results."""

    def __init__(
        self,
        config: DetectorConfig,
        diagnostic_confidence: float | None = None,
        min_query_confidence: float | None = None,
    ) -> None:
        """Load the YOLO model a single time.

        Args:
            config: Validated detector settings (model name and confidence
                threshold). If the weights are not present locally, Ultralytics
                downloads them on first use.
        """
        self._config = config
        # When set, YOLO is queried at this LOWER confidence and the configured
        # threshold is applied here instead. The pipeline still receives exactly
        # the same detections it always would, but sub-threshold boxes become
        # visible in the log -- which is the only way to tell "the bottle was
        # not detected" apart from "the bottle was detected and discarded".
        # Caveat: querying at a lower confidence lets more boxes into YOLO's own
        # NMS, so surviving boxes can differ slightly from a normal run.
        self._diagnostic_confidence = diagnostic_confidence
        # Lower floor for the YOLO query so class-specific downstream policy
        # (e.g. a 0.30 threshold for bottle-equivalent classes) can actually see
        # the detections it needs. The Detector stays class-agnostic: it only
        # lowers the query, never applies a class rule. When None, the standard
        # config.confidence_threshold is used.
        self._min_query_confidence = min_query_confidence
        # Populated each detect() call while in diagnostic mode, and read by the
        # pipeline so every stage can be reported under ONE frame number. Stays
        # None in normal operation.
        self.last_trace: DetectorTrace | None = None
        # Diagnostics only; counts calls to detect(). Used for duplicate alerts,
        # which are event-based rather than part of the per-frame trace.
        self._frame_number = 0
        _log.info("Loading YOLO model '%s'...", config.model)
        self._model = YOLO(config.model)
        _log.info("YOLO model loaded.")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single image.

        Args:
            image: A BGR image array of shape ``(height, width, 3)``.

        Returns:
            A list of :class:`Detection` objects for every object found at or
            above the configured confidence threshold. Empty if nothing is found.
        """
        self._frame_number += 1

        # verbose=False silences Ultralytics' own per-frame console output; we
        # do our own logging. conf applies the confidence threshold inside YOLO.
        if self._diagnostic_confidence is not None:
            query_confidence = self._diagnostic_confidence
        elif self._min_query_confidence is not None:
            # Query at the LOWER of the two thresholds so class-specific policy
            # downstream can see borderline detections. If the caller set the
            # min ABOVE the standard threshold (a config mistake), we keep the
            # standard threshold rather than tightening it silently.
            query_confidence = min(
                self._min_query_confidence, self._config.confidence_threshold
            )
        else:
            query_confidence = self._config.confidence_threshold
        results = self._model.predict(
            source=image,
            conf=query_confidence,
            verbose=False,
        )

        # predict() returns one Results object per input image; we passed one.
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self._log_raw_detections([], [])
            return []

        class_names = result.names  # mapping: class_id -> class label
        detections: list[Detection] = []
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=class_names.get(class_id, str(class_id)),
                )
            )

        # The effective floor mirrors the query threshold: in normal operation
        # this is a no-op because YOLO already applied it; in diagnostic mode
        # it keeps sub-threshold detections available to the trace log without
        # returning them. Class-specific policy (e.g. accepting bottle at 0.30
        # while others need 0.50) is applied downstream in the Postprocessor.
        effective_floor = (
            self._min_query_confidence
            if self._min_query_confidence is not None
            else self._config.confidence_threshold
        )
        kept = [d for d in detections if d.confidence >= effective_floor]
        self._log_raw_detections(detections, kept)
        return kept

    # ------------------------------------------------------------------
    # Diagnostics (no effect on the returned detections)
    # ------------------------------------------------------------------

    @staticmethod
    def find_duplicate_pairs(
        detections: list[Detection],
    ) -> list[tuple[int, int, float, float]]:
        """Return same-class detection pairs that look like one object seen twice.

        Args:
            detections: Detections from a single frame.

        Returns:
            ``(index_a, index_b, iou, containment)`` for each suspect pair.
        """
        suspects: list[tuple[int, int, float, float]] = []
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if detections[i].class_name != detections[j].class_name:
                    continue
                overlap = iou(detections[i].bbox, detections[j].bbox)
                inside = containment(detections[i].bbox, detections[j].bbox)
                if overlap >= _DUPLICATE_IOU or inside >= _DUPLICATE_CONTAINMENT:
                    suspects.append((i, j, overlap, inside))
        return suspects

    def _log_raw_detections(
        self, detections: list[Detection], kept: list[Detection]
    ) -> None:
        """Record this frame's YOLO output for the pipeline trace.

        In diagnostic mode the full picture -- including boxes the confidence
        threshold discarded -- is stored on ``last_trace``. The pipeline logs it
        under the authoritative frame number rather than this module printing a
        second, independently-numbered line.
        """
        if self._diagnostic_confidence is not None:
            self.last_trace = DetectorTrace(
                query_confidence=self._diagnostic_confidence,
                threshold=self._config.confidence_threshold,
                raw=list(detections),
                kept=list(kept),
            )

        duplicates = self.find_duplicate_pairs(kept)

        if _log.isEnabledFor(logging.DEBUG):
            listing = (
                " | ".join(
                    f"det{i} {d.class_name} conf={d.confidence:.2f} {d.bbox}"
                    for i, d in enumerate(kept)
                )
                or "none"
            )
            _log.debug(
                "detector frame %d: %d detection(s): %s",
                self._frame_number,
                len(kept),
                listing,
            )

        if duplicates:
            detail = " | ".join(
                f"det{i}+det{j} '{kept[i].class_name}' "
                f"iou={overlap:.2f} containment={inside:.2f} "
                f"conf={kept[i].confidence:.2f}/{kept[j].confidence:.2f}"
                for i, j, overlap, inside in duplicates
            )
            _log.info(
                "DUPLICATE DETECTION at detector frame %d (%d boxes total): %s",
                self._frame_number,
                len(kept),
                detail,
            )
