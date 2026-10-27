"""
PPO Policy Loss.

Source:
  - title: "Proximal Policy Optimization Algorithms"
    url: "https://arxiv.org/abs/1707.06347"
    year: 2017
  - title: "Upstream implementation - CleanRL ppo_continuous_action.py"
    url: "https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_continuous_action.py"
    year: null

Description:
  Exposes the clipped PPO surrogate policy objective as an atomic Teia loss node. The objective rebuilds the current Gaussian policy from `pred.policy` and applies the standard clipped ratio surrogate.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseLoss
from teia.node.net._gaussian_policy_utils import LOG_STD_MAX, LOG_STD_MIN, gaussian_distribution


class PpoPolicyLoss(BaseLoss):
    """Clipped PPO surrogate (Schulman et al. 2017) for a diagonal Gaussian policy.

        ρ = exp(log π(a|s) - log π_old(a|s))
        L = mean max(-A ρ, -A clip(ρ, 1 - ε, 1 + ε))

    Args:
        clip_coef: ``ε``.
        normalize_advantage: Standardise ``A`` over the batch.
        log_std_min: Lower clamp of ``log σ``.
        log_std_max: Upper clamp of ``log σ``.
    """

    def __init__(
        self,
        *,
        clip_coef: float = 0.2,
        normalize_advantage: bool = False,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        **_: Any,
    ) -> None:
        super().__init__()
        self.clip_coef = float(clip_coef)
        self.normalize_advantage = bool(normalize_advantage)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, mean: Tensor, log_std: Tensor, action: Tensor, old_log_prob: Tensor, advantage: Tensor) -> Tensor:
        # log π(a|s) under the current policy
        dist = gaussian_distribution(mean, log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max)
        new_log_prob = dist.log_prob(action).sum(-1)
        advantage = advantage.reshape(-1)
        old_log_prob = old_log_prob.reshape(-1)
        if self.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # ρ and the clipped surrogate
        ratio = (new_log_prob - old_log_prob).exp()
        unclipped = -advantage * ratio
        clipped = -advantage * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
        return torch.max(unclipped, clipped).mean()


__all__ = ["PpoPolicyLoss"]
