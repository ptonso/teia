"""
DQN Temporal-Difference Loss.

Source:
  - title: "Playing Atari with Deep Reinforcement Learning"
    url: "https://arxiv.org/abs/1312.5602"
    year: 2013
  - title: "Human-level control through deep reinforcement learning"
    url: "https://www.nature.com/articles/nature14236"
    year: 2015
  - title: "Upstream implementation - CleanRL dqn.py"
    url: "https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/dqn.py"
    year: null

Description:
  Computes Bellman regression from online Q-values and frozen target Q-values. The target has the form r + gamma^k(1-d) max_a Q_target(s_next, a), with optional sample weights from replay.

Adaptations:
  - Huber error by default (the clipped error of Mnih et al. 2015) where CleanRL uses squared error; `huber: false` restores it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss


class DqnLoss(BaseLoss):
    """DQN TD loss (Mnih et al. 2015), optionally n-step and importance-weighted.

        y = r + γⁿ (1 - done) max_a' Q_target(s', a')
        L = mean(w · ℓ(Q(s, a) - y)),   ℓ = Huber or ½x²

    Args:
        gamma: ``γ`` when the batch carries no ``n_step_discount``.
        huber: Huber ``ℓ`` instead of squared error.
    """

    def __init__(self, *, gamma: float = 0.99, huber: bool = True, **_: Any) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.huber = bool(huber)

    def forward(
        self,
        q: Tensor,
        action: Tensor,
        reward: Tensor,
        q_target: Tensor,
        terminated: Tensor,
        n_step_discount: Tensor | None = None,
        weight: Tensor | None = None,
    ) -> Tensor:
        # Q(s, a)
        action = action.reshape(-1).to(torch.long)
        q_pred = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        reward = reward.reshape(-1).to(q.dtype)
        terminated = terminated.reshape(-1).to(q.dtype)
        discount = (
            n_step_discount.reshape(-1).to(q.dtype)
            if n_step_discount is not None
            else torch.full_like(reward, self.gamma)
        )
        # y = r + γⁿ (1 - done) max Q_target
        next_q = q_target.detach().max(dim=-1).values
        target = reward + discount * (1.0 - terminated) * next_q

        # mean(w · ℓ(Q - y))
        if self.huber:
            per_sample = F.smooth_l1_loss(q_pred, target, reduction="none")
        else:
            per_sample = 0.5 * (q_pred - target).pow(2)
        if weight is not None:
            per_sample = per_sample * weight.reshape(-1).to(q.dtype)
        return per_sample.mean()


__all__ = ["DqnLoss"]
