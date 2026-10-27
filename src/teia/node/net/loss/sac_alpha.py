"""
SAC Temperature Loss.

Source:
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018
  - title: "Soft Actor-Critic Algorithms and Applications"
    url: "https://arxiv.org/abs/1812.05905"
    year: 2018

Description:
  Exposes the SAC entropy-temperature objective as an atomic Teia loss node. The loss adapts `log_alpha` toward the configured or inferred target entropy.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from teia.base.net import BaseLoss


class SacAlphaLoss(BaseLoss):
    """SAC temperature objective: ``L = -mean(log α · (sg(log π(a|s)) + H̄))``.

    Args:
        target_entropy: ``H̄``; ``None`` takes ``-action_dim`` from the datamodule.
    """

    def __init__(self, *, target_entropy: float | None = None, **_: Any) -> None:
        super().__init__()
        self.target_entropy = None if target_entropy is None else float(target_entropy)

    def configure_from_datamodule(self, dm: Any) -> None:
        if self.target_entropy is None:
            self.target_entropy = -float(dm.action_dim)

    def forward(self, logp_pi: Tensor, log_alpha: Tensor) -> Tensor:
        if self.target_entropy is None:
            raise ValueError("SacAlphaLoss needs `target_entropy` or a datamodule that publishes `action_dim`.")
        return -(log_alpha.reshape(()) * (logp_pi.reshape(-1).detach() + self.target_entropy)).mean()


__all__ = ["SacAlphaLoss"]
