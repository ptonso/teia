"""
SAC Actor Loss.

Source:
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018
  - title: "Soft Actor-Critic Algorithms and Applications"
    url: "https://arxiv.org/abs/1812.05905"
    year: 2018

Description:
  Exposes the reparameterized SAC policy objective as an atomic Teia loss node. The loss minimizes `alpha * log pi - min Q` using the sampled policy action.
"""

from __future__ import annotations

from torch import Tensor

from teia.base.net import BaseLoss


class SacActorLoss(BaseLoss):
    """SAC actor objective (Haarnoja et al. 2018): ``L = mean(α log π(a|s) - min(Q₁, Q₂)(s, a))`` with ``a`` reparameterized and ``α`` detached."""

    def forward(self, q1_pi: Tensor, q2_pi: Tensor, logp_pi: Tensor, log_alpha: Tensor) -> Tensor:
        alpha = log_alpha.exp().reshape(()).detach()
        min_q = q1_pi.reshape(-1).minimum(q2_pi.reshape(-1))
        return (alpha * logp_pi.reshape(-1) - min_q).mean()


__all__ = ["SacActorLoss"]
