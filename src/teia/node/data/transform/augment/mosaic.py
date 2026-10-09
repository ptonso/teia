"""Mosaic cross-sample augment — a 2×2 composite over four letterboxed items.

Source:
  - title: "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    url: "https://arxiv.org/abs/2004.10934"
    year: 2020

Description:
  Composites the primary item with three siblings (drawn through the executor-injected ``siblings``
  provider) onto a ``2·size`` canvas, emitting normalized-cxcywh boxes / keypoints / polygons relative
  to that canvas. A downstream ``RandomPerspectiveCrop`` warps the ``2·size`` canvas back to ``size``.
  The ``close_mosaic`` epoch window is owned here via a fork-shared gate; ``tick_mosaic_schedule`` is the
  ``on_train_epoch_start`` callback that pushes the trainer epoch in.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any

import torch

from teia.core.deps import require_dependency
from teia.node.data.transform.augment import ItemAugment


class Mosaic(ItemAugment):
    """2×2 mosaic of four items onto a ``2·size`` canvas (train-only, gated by ``close_mosaic``)."""

    needs_siblings = True

    def __init__(
        self, *, size: int | None = None, n: int = 4, p: float = 1.0, fill: int = 114, close_mosaic: int = 0,
    ) -> None:
        self._size = size
        self.image_size: Any = None
        self.n = int(n)
        self.p = float(p)
        self.fill = int(fill)
        self.close_mosaic = int(close_mosaic)
        self.siblings: Any = None
        self.siblings_len: int = 0
        self._epoch = mp.Value("i", 0)
        self._max_epochs = mp.Value("i", -1)

    def _side(self) -> int:
        size = self._size if self._size is not None else self.image_size
        if size is None:
            raise ValueError("Mosaic needs `size` or an injected `image_size`.")
        return int(size) if isinstance(size, int) else int(size[0])

    def set_schedule(self, max_epochs: int | None) -> None:
        with self._max_epochs.get_lock():
            self._max_epochs.value = int(max_epochs) if max_epochs is not None else -1

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def is_open(self) -> bool:
        max_epochs = self._max_epochs.value
        return not (max_epochs > 0 and self._epoch.value >= max_epochs - self.close_mosaic)

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.siblings is None or self.siblings_len <= 0 or random.random() >= self.p or not self.is_open():
            return record
        siblings = [self._sibling_record(self.siblings(random.randrange(self.siblings_len)))
                    for _ in range(self.n - 1)]
        return self._composite([record, *siblings])

    def _composite(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        require_dependency("torchvision", "Mosaic")
        from torchvision.transforms.v2 import functional as tvf

        side = self._side()
        canvas = 2 * side
        while len(records) < 4:
            records.append(records[0])
        records = records[:4]

        yc = int(random.uniform(side // 2, 2 * side - side // 2))
        xc = int(random.uniform(side // 2, 2 * side - side // 2))
        img4 = torch.full((3, canvas, canvas), self.fill, dtype=torch.uint8)

        boxes_parts: list[torch.Tensor] = []
        class_parts: list[torch.Tensor] = []
        kpt_parts: list[torch.Tensor] = []
        polys: list[torch.Tensor] = []
        has_kpts = any("keypoints" in r for r in records)
        has_polys = any("polys" in r for r in records)

        for i, rec in enumerate(records):
            img = rec["image"]
            h0, w0 = int(img.shape[-2]), int(img.shape[-1])
            ratio = side / max(h0, w0)
            nw, nh = max(1, round(w0 * ratio)), max(1, round(h0 * ratio))
            tile = tvf.resize(img, [nh, nw], antialias=True)

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                x1b, y1b = nw - (x2a - x1a), nh - (y2a - y1a)
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - nh, 0), min(xc + nw, canvas), yc
                x1b, y1b = 0, nh - (y2a - y1a)
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - nw, 0), yc, xc, min(canvas, yc + nh)
                x1b, y1b = nw - (x2a - x1a), 0
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + nw, canvas), min(canvas, yc + nh)
                x1b, y1b = 0, 0
            img4[:, y1a:y2a, x1a:x2a] = tile[:, y1b : y1b + (y2a - y1a), x1b : x1b + (x2a - x1a)]
            padw, padh = x1a - x1b, y1a - y1b

            boxes = rec.get("boxes")
            if boxes is not None and boxes.numel():
                moved = torch.stack([
                    (boxes[:, 0] * w0 * ratio + padw) / canvas,
                    (boxes[:, 1] * h0 * ratio + padh) / canvas,
                    boxes[:, 2] * w0 * ratio / canvas,
                    boxes[:, 3] * h0 * ratio / canvas,
                ], dim=1)
                boxes_parts.append(moved)
                cls = rec.get("box_cls")
                class_parts.append(cls if torch.is_tensor(cls) else torch.zeros(moved.shape[0], dtype=torch.long))

            if has_kpts:
                kpts = rec.get("keypoints")
                if kpts is not None and kpts.numel():
                    xy = torch.stack([
                        (kpts[..., 0] * w0 * ratio + padw) / canvas,
                        (kpts[..., 1] * h0 * ratio + padh) / canvas,
                    ], dim=-1)
                    kpt_parts.append(torch.cat([xy, kpts[..., 2:3]], dim=-1))

            if has_polys:
                for poly in rec.get("polys", []):
                    scaled = poly * torch.tensor([w0 * ratio, h0 * ratio], dtype=torch.float32)
                    scaled = scaled + torch.tensor([float(padw), float(padh)], dtype=torch.float32)
                    polys.append(scaled / canvas)

        out: dict[str, Any] = {"image": img4}
        out["boxes"] = torch.cat(boxes_parts, dim=0) if boxes_parts else torch.zeros((0, 4), dtype=torch.float32)
        out["box_cls"] = torch.cat(class_parts, dim=0) if class_parts else torch.zeros(0, dtype=torch.long)
        if has_kpts:
            kpt_dim = next((r["keypoints"].shape[1:] for r in records if r.get("keypoints") is not None
                            and r["keypoints"].numel()), (0, 3))
            out["keypoints"] = torch.cat(kpt_parts, dim=0) if kpt_parts else torch.zeros((0, *kpt_dim))
        if has_polys:
            out["polys"] = polys
        return out


def tick_mosaic_schedule(trainer: Any, pl_module: Any) -> None:
    """``on_train_epoch_start`` hook: push the trainer epoch/max-epochs into every ``Mosaic`` node so its
    worker-side ``is_open()`` can close the mosaic window over the last ``close_mosaic`` epochs."""
    datamodule = getattr(trainer, "datamodule", None)
    nodes = getattr(datamodule, "_nodes", {})
    for node in nodes.values():
        if isinstance(node, Mosaic):
            node.set_schedule(getattr(trainer, "max_epochs", None))
            node.set_epoch(getattr(trainer, "current_epoch", 0))
