"""Frame annotation.

Responsibility: given a frame and a list of tracks, draw bounding boxes and
labels (track ID + class name + confidence) onto a copy of the frame and return
it. Pure drawing -- no detection, no filtering, no tracking, no display.

It consumes the pipeline's framework-agnostic ``Track`` objects, so it knows
nothing about YOLO or about which tracker produced them.

Kept intentionally minimal: a single colour, no configurable styling.
"""

from __future__ import annotations

import cv2
import numpy as np

from bottlevision.detection import Track

# Fixed minimal style (BGR colours -- OpenCV's channel order).
_BOX_COLOR = (0, 255, 0)  # green
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_LABEL_MARGIN = 8  # pixels above the box for the text baseline


class Annotator:
    """Draws tracked objects onto frames."""

    def annotate(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """Return a copy of ``frame`` with each track drawn on it.

        Args:
            frame: A BGR image array of shape ``(height, width, 3)``.
            tracks: The tracked objects to draw. An empty list returns an
                unmodified copy of the frame.

        Returns:
            A new annotated BGR image array (the input frame is not mutated).
        """
        annotated = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

            label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
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
