"""``RasterizeInstanceMasks`` — letterboxed polygons → per-sample overlap + semantic masks.

Builds the overlap mask (each instance painted with its 1-based index) and the auxiliary
semantic mask (each instance painted with its class) at the ``mask_downsample_ratio`` resolution, so the
collate merely stacks per-sample maps. Faithful to the former ``InstanceSegCollator`` mask logic.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.data import Transform


def rasterize_polygons(polygons: list[list[tuple[float, float]]], *, height: int, width: int) -> torch.Tensor:
    """Fill pixel-space polygons into ``[N, H, W]`` uint8 masks (even-odd scan-line)."""
    masks = torch.zeros((len(polygons), height, width), dtype=torch.uint8)
    for index, points in enumerate(polygons):
        if len(points) < 3:
            continue
        y_min = max(0, int(min(p[1] for p in points)))
        y_max = min(height - 1, int(max(p[1] for p in points)))
        count = len(points)
        for y in range(y_min, y_max + 1):
            xs: list[float] = []
            for pi in range(count):
                x0, y0 = points[pi]
                x1, y1 = points[(pi + 1) % count]
                if ((y0 <= y < y1) or (y1 <= y < y0)) and y1 != y0:
                    xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for pi in range(0, len(xs) - 1, 2):
                left = max(0, int(xs[pi]))
                right = min(width, int(xs[pi + 1]) + 1)
                masks[index, y, left:right] = 1
    return masks


class RasterizeInstanceMasks(Transform):
    """``(item.image[3,S,S], item.polys[list], item.box_cls[N]) → (item.overlap[Hm,Wm], item.sem[Hm,Wm])``."""

    def __init__(self, *, mask_downsample_ratio: int = 4) -> None:
        self.ratio = int(mask_downsample_ratio)

    def __call__(self, image: torch.Tensor, polys: list[torch.Tensor], box_cls: torch.Tensor) -> tuple[Any, ...]:
        _, side_h, side_w = image.shape
        hm, wm = max(1, side_h // self.ratio), max(1, side_w // self.ratio)
        overlap = torch.zeros((hm, wm), dtype=torch.uint8)
        sem = torch.zeros((hm, wm), dtype=torch.float32)
        if not polys:
            return overlap, sem
        pixel_polys = [[(float(x) * wm, float(y) * hm) for x, y in poly.tolist()] for poly in polys]
        masks = rasterize_polygons(pixel_polys, height=hm, width=wm)
        cls_list = box_cls.tolist()
        for index, mask in enumerate(masks, start=1):
            pixels = mask > 0
            overlap[pixels] = index
            if index - 1 < len(cls_list):
                sem[pixels] = float(cls_list[index - 1])
        return overlap, sem
