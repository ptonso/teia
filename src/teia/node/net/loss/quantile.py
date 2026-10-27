"""
Quantile Pinball Loss.

Source:
  - title: "Regression Quantiles"
    url: "https://doi.org/10.2307/1913643"
    year: 1978

Description:
  Optimizes quantile forecasts with the pinball loss max(q e, (q - 1) e), where e is target minus prediction. The asymmetric penalty estimates conditional quantiles instead of the conditional mean.
"""


from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from teia.base.net import BaseLoss


class QuantileLoss(BaseLoss):
    """Pinball loss (Koenker & Bassett 1978), averaged over quantiles.

        e = y - ŷ_τ,   L = mean max(τ e, (τ - 1) e)

    Args:
        quantiles: Levels ``τ``, in the head's output order.
    """

    def __init__(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.quantiles = [float(q) for q in quantiles]
        self.register_buffer("_q", torch.tensor(self.quantiles), persistent=False)

    def _align(self, logits: Tensor, target: Tensor) -> Tensor:
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(-1)
        return target

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # e = y - ŷ_τ
        target = self._align(logits, target.float())
        error = target - logits

        # max(τ e, (τ - 1) e)
        q = self._q.to(logits.device)
        pinball = torch.maximum(q * error, (q - 1.0) * error)
        return pinball.mean()
