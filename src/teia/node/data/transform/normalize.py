"""``NormalizeImage`` — post-transfer uint8 → float32 scale + mean/std normalize (batch stage).

Runs in ``on_after_batch_transfer`` (GPU): the data-graph equivalent of the former datamodule GPU default
(``v2.ToDtype(scale=True)`` + ``v2.Normalize``). ImageNet stats by default.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.base.data import Transform

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizeImage(Transform):
    """uint8 → float32 normalize. ``pixel_norm``: ``imagenet`` (mean/std), ``unit`` (``[0,1]``),
    ``signed`` (``[-1,1]``). Explicit ``mean``/``std`` override ``pixel_norm``."""

    def __init__(
        self,
        *,
        mean: tuple[float, float, float] | None = _IMAGENET_MEAN,
        std: tuple[float, float, float] | None = _IMAGENET_STD,
        scale: bool = True,
        pixel_norm: str | None = None,
    ) -> None:
        self._pixel_norm = pixel_norm
        if pixel_norm in ("unit", "signed"):
            mean = std = None
        self._mean = mean
        self._std = std
        self._scale = scale

    def params(self) -> dict[str, Any]:
        return {"scale": self._scale, "mean": self._mean, "std": self._std, "pixel_norm": self._pixel_norm}

    @staticmethod
    def kernel(value: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        x = value.to(torch.float32)
        if params.get("scale", True):
            x = x / 255.0
        if params.get("pixel_norm") == "signed":
            x = x * 2.0 - 1.0
        mean, std = params.get("mean"), params.get("std")
        if mean is not None and std is not None:
            m = torch.tensor(mean, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            s = torch.tensor(std, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
            x = (x - m) / s
        return x, {}
