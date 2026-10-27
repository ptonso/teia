"""
Quantile Forecast Activation.

Source:
  - title: "Regression Quantiles"
    url: "https://doi.org/10.2307/1913643"
    year: 1978

Description:
  Formats quantile forecast outputs as prediction tensors indexed by quantile level. The activation preserves the ordered quantile structure produced by the model head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseActivation


class QuantileActivation(BaseActivation):
    """Quantile predictions, sorted so they never cross; ``params()`` publishes the levels.

    Args:
        quantiles: Levels ``τ`` the head predicts, in output order.
    """


    def __init__(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.quantiles = [float(q) for q in quantiles]

    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        return {"quantiles": torch.sort(logits, dim=-1).values}

    def params(self) -> dict[str, Any]:
        return {"levels": self.quantiles}

    @staticmethod
    def kernel(activated: dict[str, Tensor], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        del ctx
        return [{"quantiles": row.float().numpy()} for row in torch.as_tensor(activated["quantiles"]).detach().cpu()]
