"""
Tabular Regression Activation.

Source: common knowledge

Description:
  Formats Gaussian regression parameters into prediction tensors. The activation separates mean and variance-like outputs without changing the training objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseActivation


class GaussianActivation(BaseActivation):
    """Point regression: expose logits as ``{mean, std}`` with unit ``std``.

    Pairs with ``GaussianNLLLoss`` (fixed unit variance); the two-key shape keeps
    reporters/exporters consistent with probabilistic regression.
    """


    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        return {"mean": logits, "std": torch.ones_like(logits)}

    @staticmethod
    def kernel(activated: dict[str, Tensor], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        del ctx
        mean = torch.as_tensor(activated["mean"]).detach().cpu()
        std = torch.as_tensor(activated["std"]).detach().cpu()
        return [{"mean": m.float().numpy(), "std": s.float().numpy()} for m, s in zip(mean, std)]
