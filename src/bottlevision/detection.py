"""The pipeline's data contracts -- the "clean seam".

Responsibility: define the framework-agnostic shapes that flow between stations:

* :class:`Detection` -- one object seen in ONE frame (no identity).
* :class:`Track`     -- a detection that has been given a persistent identity
  across frames.

The detector converts YOLO's raw output into ``Detection`` objects; the tracker
turns those into ``Track`` objects. Because neither type mentions YOLO or
OpenCV, any station can be swapped without touching the rest of the app -- the
annotator depends on the ``Track`` *type*, not on a particular tracker.
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


@dataclass(frozen=True)
class Track:
    """A detected object carrying a persistent identity across frames.

    Attributes:
        track_id: Stable integer identity. The same physical bottle keeps the
            same ``track_id`` for as long as the tracker can follow it.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` pixel coordinates.
        confidence: Confidence of the detection that updated this track.
        class_name: Human-readable class label (e.g. ``"bottle"``).
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
