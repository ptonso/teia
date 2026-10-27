"""
PPO Value Loss.

Source:
  - title: "Proximal Policy Optimization Algorithms"
    url: "https://arxiv.org/abs/1707.06347"
    year: 2017
  - title: "Upstream implementation - CleanRL ppo_continuous_action.py"
    url: "https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_continuous_action.py"
    year: null

Description:
  Exposes PPO value regression as an atomic Teia loss node. The loss supports optional clipped value regression against the rollout's stored baseline values.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss


class ValueLoss(BaseLoss):
    """PPO value regression, optionally clipped around the old value.

        L = ½ mean max((V - R)², (V_old + clip(V - V_old, -ε, ε) - R)²)    with ``clip_coef``
        L = ½ MSE(V, R)                                                      otherwise

    Args:
        clip_coef: ``ε``; ``None`` disables clipping.
    """

    def __init__(self, *, clip_coef: float | None = None, **_: Any) -> None:
        super().__init__()
        self.clip_coef = float(clip_coef) if clip_coef is not None else None

    def forward(self, value: Tensor, return_: Tensor, old_value: Tensor | None = None) -> Tensor:
        value = value.reshape(-1)
        target = return_.reshape(-1)
        if self.clip_coef is not None and old_value is not None:
            old_value = old_value.reshape(-1)
            unclipped = (value - target).pow(2)
            clipped_value = old_value + torch.clamp(value - old_value, -self.clip_coef, self.clip_coef)
            clipped = (clipped_value - target).pow(2)
            return 0.5 * torch.max(unclipped, clipped).mean()
        return 0.5 * F.mse_loss(value, target)


__all__ = ["ValueLoss"]
