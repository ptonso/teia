"""General tensor collates — contract: per-sample stackable values, no spatial structure.

Every node here assembles one atomic ``batch.<field>`` by stacking per-sample tensors/scalars along a
new batch axis (``structure = None``). They impose no grid/geometry requirement, so they are the shared
vocabulary for *any* domain: tabular feature fan-out (``numerical``/``categorical``/``count``/…) and
class/regression targets alike route through them — a tabular ``cls`` target and a vision ``cls`` target
speak the same collate.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.data import Collate


class Stack(Collate):
    """Any per-sample value → ``batch.<field>`` ``[B, ...]`` cast to ``dtype``. The generic feature stack."""

    structure = None

    def __init__(self, *, dtype: str = "float32") -> None:
        self.dtype = dtype

    def params(self) -> dict[str, Any]:
        return {"dtype": self.dtype}

    @staticmethod
    def kernel(field_batch: list[Any], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        dtype = getattr(torch, params.get("dtype", "float32"))
        tensors = [value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=dtype) for value in field_batch]
        if not tensors:
            return torch.empty((0,), dtype=dtype), {}
        return torch.stack([tensor.to(dtype) for tensor in tensors], 0), {}


class ClassLabels(Collate):
    """``item.cls`` (scalar int, tensor or Python) → ``batch.cls`` ``[B]`` int64. The single-label target."""

    field = "cls"
    structure = None

    @staticmethod
    def kernel(field_batch: list[Any], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0,), dtype=torch.int64), {}
        return torch.stack([torch.as_tensor(t).reshape(()) for t in field_batch], 0).to(torch.int64), {}


class MultiHotClasses(Collate):
    """``item.cls`` (multi-hot ``[num_classes]``, tensor or list) → ``batch.cls`` ``[B, num_classes]`` float32."""

    field = "cls"
    structure = None

    @staticmethod
    def kernel(field_batch: list[Any], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0, 0), dtype=torch.float32), {}
        return torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in field_batch], 0), {}


class StackFactors(Collate):
    """``item.factors`` (``[F]`` float32, ``NaN`` where absent) → ``batch.labels`` ``[B, F]``."""

    field = "labels"
    structure = None

    @staticmethod
    def kernel(field_batch: list[torch.Tensor], params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        del params
        if not field_batch:
            return torch.zeros((0, 0), dtype=torch.float32), {}
        return torch.stack([t.to(torch.float32) for t in field_batch], 0), {}
