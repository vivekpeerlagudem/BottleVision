"""Object detector.

Responsibility: load the pre-trained YOLO model ONCE and run it on a single
frame, returning framework-agnostic detections. This module intentionally knows
nothing about "bottles" specifically -- class-specific policy lives in the
postprocessor.

The model is loaded in ``__init__`` and reused for every call to :meth:`detect`,
which is exactly what the real-time webcam loop needs.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from bottlevision.config import DetectorConfig
from bottlevision.detection import Detection
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)


class Detector:
    """Wraps an Ultralytics YOLO model and returns framework-agnostic results."""

    def __init__(
        self,
        config: DetectorConfig,
        min_query_confidence: float | None = None,
    ) -> None:
        """Load the YOLO model a single time.

        Args:
            config: Validated detector settings (model name and standard
                confidence threshold). If the weights are not present locally,
                Ultralytics downloads them on first use.
            min_query_confidence: Optional lower floor for the YOLO query so
                class-specific downstream policy can see borderline detections
                (e.g. bottle-equivalent classes accepted at 0.30 while everything
                else needs 0.50). The Detector stays class-agnostic: it only
                lowers the query, never applies a class rule. When ``None`` the
                standard ``config.confidence_threshold`` is used.
        """
        self._config = config
        self._min_query_confidence = min_query_confidence
        _log.info("Loading YOLO model '%s'...", config.model)
        self._model = YOLO(config.model)
        _log.info("YOLO model loaded.")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single image.

        Args:
            image: A BGR image array of shape ``(height, width, 3)``.

        Returns:
            A list of :class:`Detection` objects for every object found at or
            above the effective query threshold. Empty if nothing is found.
        """
        # Query at the LOWER of the two thresholds so class-specific policy
        # downstream can see borderline detections. Never tighten the query
        # above the configured threshold by accident.
        if self._min_query_confidence is not None:
            query_confidence = min(
                self._min_query_confidence, self._config.confidence_threshold
            )
        else:
            query_confidence = self._config.confidence_threshold

        # verbose=False silences Ultralytics' own per-frame console output.
        results = self._model.predict(
            source=image,
            conf=query_confidence,
            verbose=False,
        )

        # predict() returns one Results object per input image; we passed one.
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
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
        return detections
