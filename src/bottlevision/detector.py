"""Object detector.

Responsibility: load the pre-trained YOLO model ONCE and run it on a single
frame, returning raw detections (bounding box, class, confidence) for EVERY
object it finds. This module intentionally knows nothing about "bottles"
specifically -- filtering happens later, in post-processing.

The model is loaded in ``__init__`` and reused for every call to :meth:`detect`,
which is exactly what the real-time webcam loop will need.
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

    def __init__(self, config: DetectorConfig) -> None:
        """Load the YOLO model a single time.

        Args:
            config: Validated detector settings (model name and confidence
                threshold). If the weights are not present locally, Ultralytics
                downloads them on first use.
        """
        self._config = config
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
        # verbose=False silences Ultralytics' own per-frame console output; we
        # do our own logging. conf applies the confidence threshold inside YOLO.
        results = self._model.predict(
            source=image,
            conf=self._config.confidence_threshold,
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
