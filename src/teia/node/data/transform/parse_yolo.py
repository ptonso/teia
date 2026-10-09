"""``ParseYoloDet`` — parse a YOLO ``.txt`` label file into normalized boxes + class indices.

Native YOLO detection parser: each line is ``cls cx cy w h`` (normalized
to ``[0,1]``). Emits ``item.boxes`` ``[N,4]`` normalized ``cxcywh`` float32 and ``item.box_cls`` ``[N]``
int64. A missing label (``None`` from a ``left`` join) or empty file yields empty tensors. num_classes is
published by a separate meta node / data.yaml so parse stays per-item.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from teia.base.data import Transform


class ParseYoloDet(Transform):
    """``item.label_path → (item.boxes[N,4] cxcywh-norm, item.box_cls[N])``."""

    def __call__(self, label_path: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        boxes: list[list[float]] = []
        classes: list[int] = []
        if label_path is not None and Path(label_path).is_file():
            for line in Path(label_path).read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                classes.append(int(float(parts[0])))
                boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        box_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        cls_tensor = torch.tensor(classes, dtype=torch.int64) if classes else torch.zeros((0,), dtype=torch.int64)
        return box_tensor, cls_tensor


class ParseYoloPose(Transform):
    """``item.label_path → (item.boxes_raw[N,4], item.box_cls[N], item.kpts_raw[N,K,3])``.

    YOLO pose line: ``cls cx cy w h (x y v)*K`` (all normalized). Native YOLO pose parser.
    """

    def __call__(self, label_path: str | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes: list[list[float]] = []
        classes: list[int] = []
        kpts: list[list[list[float]]] = []
        if label_path is not None and Path(label_path).is_file():
            for line in Path(label_path).read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                classes.append(int(float(parts[0])))
                boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                rest = [float(v) for v in parts[5:]]
                kpts.append([rest[i : i + 3] for i in range(0, len(rest) - len(rest) % 3, 3)])
        n = len(boxes)
        k = len(kpts[0]) if kpts and kpts[0] else 0
        box_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        cls_tensor = torch.tensor(classes, dtype=torch.int64) if classes else torch.zeros((0,), dtype=torch.int64)
        kpt_tensor = torch.tensor(kpts, dtype=torch.float32) if (n and k) else torch.zeros((n, max(k, 1), 3), dtype=torch.float32)
        return box_tensor, cls_tensor, kpt_tensor


class ParseYoloSeg(Transform):
    """``item.label_path → (item.boxes_raw[N,4], item.box_cls[N], item.polys_raw: list[Tensor[K,2]])``.

    YOLO-seg line: ``cls x1 y1 x2 y2 ...`` (normalized polygon). The box is the polygon's bounding box
    (``cxcywh``). Native YOLO segmentation parser.
    """

    def __call__(self, label_path: str | None) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        boxes: list[list[float]] = []
        classes: list[int] = []
        polys: list[torch.Tensor] = []
        if label_path is not None and Path(label_path).is_file():
            for line in Path(label_path).read_text().splitlines():
                parts = line.split()
                if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                    continue
                coords = [float(v) for v in parts[1:]]
                pts = torch.tensor([[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)], dtype=torch.float32)
                if pts.shape[0] < 3:
                    continue
                classes.append(int(float(parts[0])))
                polys.append(pts)
                x1, y1 = pts.min(0).values.tolist()
                x2, y2 = pts.max(0).values.tolist()
                boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        box_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        cls_tensor = torch.tensor(classes, dtype=torch.int64) if classes else torch.zeros((0,), dtype=torch.int64)
        return box_tensor, cls_tensor, polys
