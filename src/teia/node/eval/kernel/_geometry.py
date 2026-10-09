"""
Polygon and box geometry kernels.

Source:
  - title: "Reentrant Polygon Clipping"
    url: "https://doi.org/10.1145/360767.360802"
    year: 1974

Description:
  Box and polygon math (``bbox_iou``, ``convex_polygon_intersection``, ``polygon_area``,
  ``polygon_iou``), self-contained so ``teia.node.eval`` does not depend on ``teia.core``. Not a
  component itself (no ``_target_``, no ``in``/``out``); the four kernel modules in this package
  import from it.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def polygon_area(points: list[Point]) -> float:
    """Absolute area of a simple polygon by the shoelace formula. Degenerate inputs give 0."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def _cross(o: Point, a: Point, b: Point) -> float:
    return ((a[0] - o[0]) * (b[1] - o[1])) - ((a[1] - o[1]) * (b[0] - o[0]))


def _line_intersection(a1: Point, a2: Point, b1: Point, b2: Point) -> Point:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denom) < 1e-12:
        return a2
    det1 = (x1 * y2) - (y1 * x2)
    det2 = (x3 * y4) - (y3 * x4)
    px = ((det1 * (x3 - x4)) - ((x1 - x2) * det2)) / denom
    py = ((det1 * (y3 - y4)) - ((y1 - y2) * det2)) / denom
    return px, py


def convex_polygon_intersection(subject: list[Point], clip: list[Point]) -> list[Point]:
    """Sutherland-Hodgman clip. ``clip`` must be convex; ``subject`` may be any simple polygon,
    but the result is exact only when both are convex."""
    output = list(subject)
    if len(output) < 3 or len(clip) < 3:
        return []
    for index, clip_end in enumerate(clip):
        clip_start = clip[index - 1]
        input_list = list(output)
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            e_inside = _cross(clip_start, clip_end, e) >= 0
            s_inside = _cross(clip_start, clip_end, s) >= 0
            if e_inside:
                if not s_inside:
                    output.append(_line_intersection(s, e, clip_start, clip_end))
                output.append(e)
            elif s_inside:
                output.append(_line_intersection(s, e, clip_start, clip_end))
            s = e
    return output


def polygon_iou(a: list[Point], b: list[Point]) -> float:
    """Intersection-over-union of two convex polygons, via analytic clipping rather than
    rasterization, so it carries no discretization error."""
    inter_poly = convex_polygon_intersection(a, b)
    inter_area = polygon_area(inter_poly)
    union = polygon_area(a) + polygon_area(b) - inter_area
    return inter_area / union if union > 0 else 0.0


def bbox_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Centre-form ``(cx, cy, w, h)`` to corner-form ``(x1, y1, x2, y2)``."""
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def bbox_area(xyxy: tuple[float, float, float, float]) -> float:
    """Area of a corner-form box, clamped at 0 for inverted coordinates."""
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two axis-aligned corner-form boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def rotated_box_corners(cx: float, cy: float, w: float, h: float, angle: float) -> list[Point]:
    """Four corners of a ``(cx, cy, w, h, angle)`` rotated box; ``angle`` in radians,
    counter-clockwise, matching the ``item.obb`` and OBB val-metric decode convention."""
    half_w, half_h = w / 2.0, h / 2.0
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]
