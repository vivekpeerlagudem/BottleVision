"""Frame annotation.

Responsibility: given a frame and a list of detections, draw bounding boxes and
labels (class name + confidence) onto a copy of the frame and return it. Pure
drawing -- no detection, no filtering, no display. It consumes the pipeline's
framework-agnostic ``Detection`` objects, so it knows nothing about YOLO.

Kept intentionally minimal for M3: a single colour, no configurable styling.
Styling from configuration is a later milestone.
"""

from __future__ import annotations

import cv2
import numpy as np

from bottlevision.detection import Detection

# Fixed minimal style (BGR colours -- OpenCV's channel order).
_BOX_COLOR = (0, 255, 0)  # green
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_LABEL_MARGIN = 8  # pixels above the box for the text baseline


class Annotator:
    """Draws detections onto frames."""

    def annotate(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Return a copy of ``frame`` with each detection drawn on it.

        Args:
            frame: A BGR image array of shape ``(height, width, 3)``.
            detections: The objects to draw. An empty list returns an
                unmodified copy of the frame.

        Returns:
            A new annotated BGR image array (the input frame is not mutated).
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - _LABEL_MARGIN)),
                _FONT,
                _FONT_SCALE,
                _BOX_COLOR,
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )
        return annotated
