"""Random affine / perspective crop-warp over the item workspace.

Source: common knowledge

Description:
  Builds a composed ``T @ S @ R @ P @ C`` forward warp mapping the (letterboxed or mosaic) input canvas
  to a square ``size`` output, grid-samples the image, and applies the same matrix to every modality.
  Coordinates enter and leave normalized: points are scaled to input pixels, warped, then divided by the
  output side. Degenerate instances (tiny side, extreme aspect, or area shrunk past ``area_thr``) are
  dropped across ``item.boxes`` / ``item.box_cls`` / ``item.keypoints`` / ``item.polys`` together.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F

from teia.node.data.transform.augment import ItemAugment


def _rand(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def _pair(value: Any) -> tuple[float, float]:
    if value is None:
        return (0.0, 0.0)
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    seq = list(value)
    return (float(seq[0]), float(seq[0])) if len(seq) == 1 else (float(seq[0]), float(seq[1]))


def _affine_matrix(
    *, degrees: float, translate: tuple[float, float], scale: Any, shear: float, perspective: float,
    in_w: int, in_h: int, out_w: int, out_h: int,
) -> torch.Tensor:
    center = torch.eye(3, dtype=torch.float64)
    center[0, 2] = -in_w / 2.0
    center[1, 2] = -in_h / 2.0

    persp = torch.eye(3, dtype=torch.float64)
    persp[2, 0] = _rand(-perspective, perspective)
    persp[2, 1] = _rand(-perspective, perspective)

    angle = _rand(-degrees, degrees)
    if isinstance(scale, (tuple, list)):
        factor = _rand(float(scale[0]), float(scale[1]))
    else:
        factor = _rand(1.0 - float(scale), 1.0 + float(scale))
    rad = math.radians(angle)
    alpha, beta = factor * math.cos(rad), factor * math.sin(rad)
    rotate = torch.eye(3, dtype=torch.float64)
    rotate[0, 0], rotate[0, 1] = alpha, beta
    rotate[1, 0], rotate[1, 1] = -beta, alpha

    shear_m = torch.eye(3, dtype=torch.float64)
    shear_m[0, 1] = math.tan(math.radians(_rand(-shear, shear)))
    shear_m[1, 0] = math.tan(math.radians(_rand(-shear, shear)))

    translate_m = torch.eye(3, dtype=torch.float64)
    translate_m[0, 2] = _rand(0.5 - translate[0], 0.5 + translate[0]) * out_w
    translate_m[1, 2] = _rand(0.5 - translate[1], 0.5 + translate[1]) * out_h

    return translate_m @ shear_m @ rotate @ persp @ center


def _sampling_grid(m_inv: torch.Tensor, in_w: int, in_h: int, out_w: int, out_h: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(out_h, dtype=torch.float64),
        torch.arange(out_w, dtype=torch.float64),
        indexing="ij",
    )
    dst = torch.stack([xs, ys, torch.ones_like(xs)], dim=0).reshape(3, -1)
    src = m_inv @ dst
    w = src[2].clamp_min(1e-8)
    gx = (src[0] / w + 0.5) * 2.0 / in_w - 1.0
    gy = (src[1] / w + 0.5) * 2.0 / in_h - 1.0
    return torch.stack([gx, gy], dim=-1).reshape(1, out_h, out_w, 2).to(torch.float32)


def _transform_points(m: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    if pts.numel() == 0:
        return pts
    homogeneous = torch.cat([pts.to(torch.float64), torch.ones((pts.shape[0], 1), dtype=torch.float64)], dim=1)
    projected = homogeneous @ m.T
    return (projected[:, :2] / projected[:, 2:3].clamp_min(1e-8)).to(torch.float32)


def _warp_image(image: torch.Tensor, grid: torch.Tensor, fill: int) -> torch.Tensor:
    centered = image.to(torch.float32).unsqueeze(0) - float(fill)
    out = F.grid_sample(centered, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return (out.squeeze(0) + float(fill)).clamp_(0.0, 255.0).round_().to(torch.uint8)


class RandomPerspectiveCrop(ItemAugment):
    """Warp the input canvas to a square ``size`` across every wired modality, dropping degenerate rows."""

    def __init__(
        self, *, size: int | None = None, degrees: float = 0.0, translate: Any = (0.1, 0.1),
        scale: Any = (0.5, 1.5), shear: float = 0.0, perspective: float = 0.0, fill: int = 114,
        area_thr: float = 0.1,
    ) -> None:
        self._size = size
        self.image_size: Any = None
        self.degrees = float(degrees)
        self.translate = _pair(translate)
        if isinstance(scale, (int, float)):
            self.scale: Any = float(scale)
        else:
            vals = list(scale)
            self.scale = (float(vals[0]), float(vals[1])) if len(vals) >= 2 else float(vals[0])
        self.shear = float(shear)
        self.perspective = float(perspective)
        self.fill = int(fill)
        self.area_thr = float(area_thr)

    def _side(self) -> int:
        size = self._size if self._size is not None else self.image_size
        if size is None:
            raise ValueError("RandomPerspectiveCrop needs `size` or an injected `image_size`.")
        return int(size) if isinstance(size, int) else int(size[0])

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        image = record["image"]
        in_h, in_w = int(image.shape[-2]), int(image.shape[-1])
        out = self._side()

        m = _affine_matrix(
            degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear,
            perspective=self.perspective, in_w=in_w, in_h=in_h, out_w=out, out_h=out,
        )
        grid = _sampling_grid(torch.linalg.inv(m), in_w, in_h, out, out)
        record["image"] = _warp_image(image, grid, self.fill)

        boxes = record.get("boxes")
        polys = record.get("polys")
        keypoints = record.get("keypoints")
        has_boxes = boxes is not None and boxes.numel() > 0
        if not has_boxes:
            return record

        pre_wh = torch.stack([boxes[:, 2] * in_w, boxes[:, 3] * in_h], dim=1)

        if polys is not None:
            new_polys: list[torch.Tensor] = []
            derived = []
            for poly in polys:
                pts = poly.to(torch.float32) * torch.tensor([in_w, in_h], dtype=torch.float32)
                warped = _transform_points(m, pts)
                warped[:, 0].clamp_(0, out)
                warped[:, 1].clamp_(0, out)
                new_polys.append(warped / out)
                if warped.shape[0]:
                    derived.append([float(warped[:, 0].min()), float(warped[:, 1].min()),
                                    float(warped[:, 0].max()), float(warped[:, 1].max())])
                else:
                    derived.append([0.0, 0.0, 0.0, 0.0])
            record["polys"] = new_polys
            xyxy = torch.tensor(derived, dtype=torch.float32) / out
        else:
            cx, cy, w, h = boxes.unbind(1)
            x1, y1 = (cx - w / 2) * in_w, (cy - h / 2) * in_h
            x2, y2 = (cx + w / 2) * in_w, (cy + h / 2) * in_h
            corners = torch.stack([
                torch.stack([x1, y1], 1), torch.stack([x2, y1], 1),
                torch.stack([x2, y2], 1), torch.stack([x1, y2], 1),
            ], dim=1).reshape(-1, 2)
            warped = _transform_points(m, corners).reshape(-1, 4, 2)
            nx1 = warped[:, :, 0].min(1).values.clamp(0, out)
            ny1 = warped[:, :, 1].min(1).values.clamp(0, out)
            nx2 = warped[:, :, 0].max(1).values.clamp(0, out)
            ny2 = warped[:, :, 1].max(1).values.clamp(0, out)
            xyxy = torch.stack([nx1, ny1, nx2, ny2], dim=1) / out

        new_w = (xyxy[:, 2] - xyxy[:, 0]) * out
        new_h = (xyxy[:, 3] - xyxy[:, 1]) * out
        record["boxes"] = torch.stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1],
        ], dim=1)

        if keypoints is not None and keypoints.numel():
            kxy = keypoints[..., :2] * torch.tensor([in_w, in_h], dtype=torch.float32)
            flat = _transform_points(m, kxy.reshape(-1, 2)).reshape(keypoints.shape[0], -1, 2)
            inside = (flat[..., 0] >= 0) & (flat[..., 0] <= out) & (flat[..., 1] >= 0) & (flat[..., 1] <= out)
            vis = keypoints[..., 2] * inside.to(keypoints.dtype)
            record["keypoints"] = torch.cat([flat / out, vis.unsqueeze(-1)], dim=-1)

        eps = 1e-9
        aspect = torch.maximum(new_w / (new_h + eps), new_h / (new_w + eps))
        keep = (new_w > 2) & (new_h > 2) & (aspect < 100)
        keep &= (new_w * new_h) / (pre_wh[:, 0] * pre_wh[:, 1] + eps) > self.area_thr
        self._filter(record, keep)
        return record

    @staticmethod
    def _filter(record: dict[str, Any], keep: torch.Tensor) -> None:
        n = int(keep.shape[0])
        record["boxes"] = record["boxes"][keep]
        for key in ("box_cls", "keypoints"):
            value = record.get(key)
            if torch.is_tensor(value) and value.shape[0] == n:
                record[key] = value[keep]
        polys = record.get("polys")
        if polys is not None and len(polys) == n:
            record["polys"] = [poly for poly, flag in zip(polys, keep.tolist()) if flag]
