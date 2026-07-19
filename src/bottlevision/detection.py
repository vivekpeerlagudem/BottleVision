"""The detection data structure -- the "clean seam" of the pipeline.

Responsibility: define the single, framework-agnostic shape that describes one
detected object. The detector converts YOLO's raw output into these objects, and
every downstream station (post-processing, annotation) consumes them. Because
this type mentions nothing about YOLO or OpenCV, the detector could be swapped
for a different model without touching the rest of the app.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object.

    Attributes:
        bbox: Bounding box as ``(x1, y1, x2, y2)`` pixel coordinates, where
            ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` the bottom-right.
        confidence: Model confidence in ``[0.0, 1.0]``.
        class_id: Integer class index as defined by the model.
        class_name: Human-readable class label (e.g. ``"bottle"``).
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
