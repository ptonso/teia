"""``ObbFromPolys`` — minimum-area oriented box ``(cx,cy,w,h,angle)`` from a polygon.

Native form of the former ``ObbCollator._polygon4_to_xywhr``. Runs after letterbox on the square canvas
(x,y normalized by the same side), so the rotation angle is metric-correct. Emits ``item.obb[N,5]``.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import math

import torch

from teia.base.data import Transform


def _poly_to_xywhr(pts: list[tuple[float, float]]) -> tuple[float, float, float, float, float]:
    if len(pts) < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in pts[:4]]
    ys = [p[1] for p in pts[:4]]
    cx, cy = sum(xs) / 4, sum(ys) / 4
    best_area, best = float("inf"), (0.0, 0.0, 0.0)
    for i in range(4):
        j = (i + 1) % 4
        angle = math.atan2(ys[j] - ys[i], xs[j] - xs[i])
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        rx = [(x - cx) * cos_a - (y - cy) * sin_a for x, y in zip(xs, ys)]
        ry = [(x - cx) * sin_a + (y - cy) * cos_a for x, y in zip(xs, ys)]
        w, h = max(rx) - min(rx), max(ry) - min(ry)
        if w * h < best_area:
            best_area, best = w * h, (w, h, angle)
    w, h, angle = best
    return cx, cy, w, h, float(angle)


class ObbFromPolys(Transform):
    """``item.polys (list[Tensor[K,2]] normalized) → item.obb[N,5]`` ``(cx,cy,w,h,angle)``."""

    def __call__(self, polys: list[torch.Tensor]) -> torch.Tensor:
        rows = [_poly_to_xywhr([tuple(p) for p in poly.tolist()]) for poly in polys]
        return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5), dtype=torch.float32)
