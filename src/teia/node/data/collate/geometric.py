"""Geometric collates — contract: requires grid / per-instance spatial geometry.

These nodes only make sense on data carrying a spatial grid (images, dense masks) or ragged per-instance
image-space geometry (boxes, keypoints painted into an image). They cannot apply to flat tabular data, so
they declare ``requires_structure = "grid"`` (or emit ``structure = "grid"`` directly) and the executor
rejects them in a graph that has no grid producer — a build-time fail-fast for a mis-wired recipe.

Two families:
- **grid stacks** (``structure = "grid"``): stack per-sample maps → ``[B, ...]`` (images, semantic /
  overlap masks).
- **ragged concat** (detection): flat-concat variable-length per-image instances with a ``batch_idx``
  bookkeeping column; boxes stay normalized ``cxcywh`` (the detection loss contract).

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.data import Collate


class StackImages(Collate):
    """``item.image`` (uint8 ``[3,H,W]``) → ``batch.image`` ``[B,3,H,W]``. Float conversion is post-transfer."""

    field = "image"
    structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0, 3, 0, 0), dtype=torch.uint8), {}
        return torch.stack(field_batch, 0), {}


class SemanticMasks(Collate):
    """``item.mask`` (int64 ``[H,W]``) → ``batch.masks`` ``[B,H,W]`` (dense semantic target)."""

    field = "masks"
    structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0, 0, 0), dtype=torch.int64), {}
        return torch.stack([m.to(torch.int64) for m in field_batch], 0), {}


class OverlapMasks(Collate):
    """``item.overlap[Hm,Wm] → batch.masks[B,Hm,Wm]`` uint8 (overlap masks)."""

    field = "masks"
    structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0, 0, 0), dtype=torch.uint8), {}
        return torch.stack([m.to(torch.uint8) for m in field_batch], 0), {}


class SemMasks(Collate):
    """``item.sem[Hm,Wm] → batch.sem_masks[B,Hm,Wm]`` float32 (auxiliary semantic masks)."""

    field = "sem_masks"
    structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0, 0, 0), dtype=torch.float32), {}
        return torch.stack([m.to(torch.float32) for m in field_batch], 0), {}


class ConcatBoxes(Collate):
    """``item.boxes[Ni,D] → batch.bboxes[sum Ni, D]`` (flat-concat). ``box_dim`` = 4 (axis-aligned) or 5 (obb)."""

    field = "bboxes"
    structure = None
    requires_structure = "grid"

    def __init__(self, *, box_dim: int = 4) -> None:
        self.box_dim = int(box_dim)

    def params(self) -> dict[str, Any]:
        return {"box_dim": self.box_dim}

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        parts = [b for b in field_batch if b is not None and b.numel()]
        box_dim = int(params.get("box_dim", 4))
        result = torch.cat(parts, 0) if parts else torch.zeros((0, box_dim), dtype=torch.float32)
        return result, {}


class ConcatClasses(Collate):
    """``item.box_cls[Ni] → batch.cls[sum Ni, 1]`` float32 (detection losses expect a column)."""

    field = "cls"
    structure = None
    requires_structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        parts = [c.reshape(-1, 1).to(torch.float32) for c in field_batch if c is not None and c.numel()]
        result = torch.cat(parts, 0) if parts else torch.zeros((0, 1), dtype=torch.float32)
        return result, {}


class BatchIndex(Collate):
    """``item.boxes (per image) → batch.batch_idx[sum Ni, 1]`` int64 (image index per box)."""

    field = "batch_idx"
    structure = None
    requires_structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        parts = [
            torch.full((int(b.shape[0]), 1), i, dtype=torch.int64)
            for i, b in enumerate(field_batch)
            if b is not None
        ]
        parts = [p for p in parts if p.numel()]
        result = torch.cat(parts, 0) if parts else torch.zeros((0, 1), dtype=torch.int64)
        return result, {}


class ConcatKeypoints(Collate):
    """``item.keypoints[Ni,K,3] → batch.keypoints[sum Ni, K, 3]`` (flat-concat, normalized xy + vis)."""

    field = "keypoints"
    structure = None
    requires_structure = "grid"

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        parts = [k for k in field_batch if k is not None and k.numel()]
        if not parts:
            return torch.zeros((0, 1, 3), dtype=torch.float32), {}
        k = max(p.shape[1] for p in parts)
        return torch.cat([p for p in parts if p.shape[1] == k], 0), {}


class ListField(Collate):
    """Pass per-sample values (``ori_shape``/``path``/``ratio_pad``) through as a list (JIT-exempt)."""

    field = "path"  # overridden per-node via the collate ``out`` key (batch.<field>)
    structure = None
    requires_structure = "grid"

    @staticmethod
    def kernel(field_batch: list[Any], params: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        del params
        return list(field_batch), {}
