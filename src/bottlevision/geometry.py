"""Pure bounding-box geometry.

Responsibility: the small set of box calculations the tracker uses. Everything
here is a pure function of plain tuples -- no OpenCV, no model, no state.
"""

from __future__ import annotations

BBox = tuple[int, int, int, int]


def iou(box_a: BBox, box_b: BBox) -> float:
    """Return the Intersection over Union of two boxes.

    1.0 for identical boxes, 0.0 for boxes that do not touch. Note that IoU
    carries no information once the boxes separate -- every non-overlapping
    pair scores exactly 0.0, which is why the tracker also matches on centre
    distance.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center(box: BBox) -> tuple[float, float]:
    """Return the ``(x, y)`` centre point of a box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def mean_side(box: BBox) -> float:
    """Return the average of a box's width and height (its rough scale)."""
    x1, y1, x2, y2 = box
    return ((x2 - x1) + (y2 - y1)) / 2.0
