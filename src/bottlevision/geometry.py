"""Pure bounding-box geometry.

Responsibility: the small set of box calculations shared by more than one
pipeline station. Keeping them here means the detector and the tracker measure
overlap with exactly the same definitions, so their logs are comparable.

Everything here is a pure function of plain tuples -- no OpenCV, no model, no
state.
"""

from __future__ import annotations

BBox = tuple[int, int, int, int]


def area(box: BBox) -> int:
    """Return the pixel area of a box (0 if degenerate)."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection(box_a: BBox, box_b: BBox) -> int:
    """Return the overlapping pixel area of two boxes (0 if disjoint)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0
    return width * height


def iou(box_a: BBox, box_b: BBox) -> float:
    """Return the Intersection over Union of two boxes.

    1.0 for identical boxes, 0.0 for boxes that do not touch. Note that IoU
    carries no information once the boxes separate -- every non-overlapping
    pair scores exactly 0.0.
    """
    overlap = intersection(box_a, box_b)
    if overlap == 0:
        return 0.0
    union = area(box_a) + area(box_b) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def containment(box_a: BBox, box_b: BBox) -> float:
    """Return overlap as a fraction of the SMALLER box's area.

    This catches nested boxes that IoU under-reports: a detector can emit one
    large box for a bottle and a second small box covering only part of it. Such
    a pair may score a modest IoU while the small box is almost entirely inside
    the large one -- containment near 1.0 -- which is still one physical object
    detected twice.
    """
    overlap = intersection(box_a, box_b)
    if overlap == 0:
        return 0.0
    smaller = min(area(box_a), area(box_b))
    if smaller <= 0:
        return 0.0
    return overlap / smaller


def center(box: BBox) -> tuple[float, float]:
    """Return the ``(x, y)`` centre point of a box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def mean_side(box: BBox) -> float:
    """Return the average of a box's width and height (its rough scale)."""
    x1, y1, x2, y2 = box
    return ((x2 - x1) + (y2 - y1)) / 2.0
