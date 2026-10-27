"""
Squashed Gaussian Sampler Node.

Source:
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018

Description:
  Exposes SAC reparameterized action sampling as an atomic Teia node. The node transforms `pred.mean` and `pred.log_std` into sampled policy actions and their corrected log probabilities for critic and alpha losses.
"""

from __future__ import annotations

from torch import Tensor

from teia.base.net import TeiaNode
from teia.node.net._gaussian_policy_utils import LOG_STD_MAX, LOG_STD_MIN, squashed_gaussian_sample


class SquashedGaussianSample(TeiaNode):
    """Tanh-squashed Gaussian sample (Haarnoja et al. 2018, App. C). Returns ``(a, log π(a|s))``.

        u = μ + σ ε,   a = tanh(u)
        log π(a|s) = log N(u; μ, σ) - Σ log(1 - tanh²(u))

    The action is mapped from ``[-1, 1]`` to ``[low, high]``; ``log π`` stays on the ``[-1, 1]`` variable.

    Args:
        log_std_min: Lower clamp of ``log σ``.
        log_std_max: Upper clamp of ``log σ``.
        low: Lower action bound (scalar or one per dimension).
        high: Upper action bound.
    """

    def build_module(
        self,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        low: float | list[float] = -1.0,
        high: float | list[float] = 1.0,
        **_: object,
    ) -> None:
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.low, self.high = low, high

    def forward(self, mean: Tensor | None, log_std: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
        if mean is None:
            return None, None
        return squashed_gaussian_sample(
            mean, log_std, deterministic=False, log_std_min=self.log_std_min, log_std_max=self.log_std_max, low=self.low, high=self.high
        )


__all__ = ["SquashedGaussianSample"]
