"""
SAC Critic Loss.

Source:
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018
  - title: "Soft Actor-Critic Algorithms and Applications"
    url: "https://arxiv.org/abs/1812.05905"
    year: 2018

Description:
  Exposes the twin soft-Q Bellman regression objective as an atomic Teia loss node. The target follows the standard soft Bellman backup with entropy correction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseLoss


class SacCriticLoss(BaseLoss):
    """SAC twin critic regression (Haarnoja et al. 2018).

        y = r + γⁿ (1 - done) (min(Q̄₁, Q̄₂)(s', a') - α log π(a'|s'))
        L = MSE(Q₁, sg(y)) + MSE(Q₂, sg(y))
    """

    def forward(
        self,
        q1_buf: Tensor,
        q2_buf: Tensor,
        reward: Tensor,
        q1_targ: Tensor,
        q2_targ: Tensor,
        next_logp: Tensor,
        terminated: Tensor,
        n_step_discount: Tensor,
        log_alpha: Tensor,
    ) -> Tensor:
        # y = r + γⁿ (1 - done) (min Q̄ - α log π)
        alpha = log_alpha.exp().reshape(()).detach()
        reward = reward.reshape(-1)
        terminated = terminated.reshape(-1).to(reward.dtype)
        discount = n_step_discount.reshape(-1).to(reward.dtype)
        min_q_targ = torch.min(q1_targ, q2_targ).reshape(-1)
        soft_targ = min_q_targ - alpha * next_logp.reshape(-1)
        target = (reward + discount * (1.0 - terminated) * soft_targ).detach()

        # MSE of both critics to y
        q1 = q1_buf.reshape(-1)
        q2 = q2_buf.reshape(-1)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)


__all__ = ["SacCriticLoss"]
