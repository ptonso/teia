"""
Reconstruction Activation.

Source:
  - title: "Auto-Encoding Variational Bayes"
    url: "https://arxiv.org/abs/1312.6114"
    year: 2013

Description:
  Maps decoder logits to the mean image, one value per pixel, in the pixel space of the task. The kernel exports each image in [0, 1], whatever the pixel space.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseActivation
from teia.node.net._pixel_space import squash


class ReconstructActivation(BaseActivation):
    """Reconstruction logits to an image in the task ``pixel_norm`` space: ``x̂ = σ(z)`` for ``unit``, ``tanh(z)`` for ``signed``.

    Args:
        strategy: ``unit`` ([0, 1]) or ``signed`` ([-1, 1]).
    """

    def __init__(self, strategy: str = "unit") -> None:
        super().__init__()
        self.strategy = strategy

    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        return {"reconstructed": squash(logits, self.strategy)}

    def params(self) -> dict[str, Any]:
        return {"strategy": self.strategy}

    @staticmethod
    def kernel(activated: dict[str, Tensor], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Export unrolls this function into the exported pipeline, so it calls no helper of this package.
        image = torch.as_tensor(activated["reconstructed"]).detach().cpu()
        image = (image + 1.0) * 0.5 if ctx["strategy"] == "signed" else image
        return [{"image": row} for row in image]
