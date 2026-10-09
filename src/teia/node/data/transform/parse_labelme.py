"""``ParseLabelme`` — parse a LabelMe JSON sidecar into image path + normalized polygons + labels.

Native LabelMe parser: resolves ``imagePath`` against the JSON's directory, reads
``polygon`` shapes, normalizes points by the image size, and derives each polygon's axis-aligned bounding
box (``cxcywh``) so the shared letterbox path applies. Class strings are indexed by ``NameListEncode``.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from teia.base.data import Transform


class ParseLabelme(Transform):
    """``item.json_path → (item.img_path, item.boxes_raw[N,4], item.poly_labels: list[str], item.polys_raw: list[Tensor[K,2]])``."""

    def __call__(self, json_path: str) -> tuple[Any, ...]:
        path = Path(json_path)
        payload = json.loads(path.read_text())
        image_field = payload.get("imagePath")
        image_path = (path.parent / image_field).resolve() if image_field else None
        if image_path is not None and not image_path.exists():
            alt = (path.parent / "images" / Path(image_field).name).resolve()
            image_path = alt if alt.exists() else image_path
        w = float(payload.get("imageWidth") or 1)
        h = float(payload.get("imageHeight") or 1)
        boxes: list[list[float]] = []
        labels: list[str] = []
        polys: list[torch.Tensor] = []
        for shape in payload.get("shapes", []):
            if not isinstance(shape, dict) or shape.get("shape_type") != "polygon":
                continue
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            pts = torch.tensor([[float(x) / w, float(y) / h] for x, y in points], dtype=torch.float32)
            polys.append(pts)
            labels.append(str(shape.get("label", "")))
            x1, y1 = pts.min(0).values.tolist()
            x2, y2 = pts.max(0).values.tolist()
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        box_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        return str(image_path), box_tensor, labels, polys
