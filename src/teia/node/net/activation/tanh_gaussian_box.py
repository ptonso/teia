"""
Tanh Gaussian Box Activation.

Source:
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018

Description:
  Exposes bounded SAC Gaussian action activation as an atomic Teia activation node. The activation produces deterministic mean actions for export and tanh-squashed stochastic samples for collection.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from teia.base.net import ActionActivation, ActionSample
from teia.node.net._gaussian_policy_utils import LOG_STD_MAX, LOG_STD_MIN, affine_range, squashed_gaussian_sample


class TanhGaussianBoxActivation(ActionActivation):
    """Tanh-squashed Gaussian policy over a bounded Box action space (SAC).

        a = tanh(μ + σ ε)      sample;  tanh(μ) when deterministic and in the traced route

    The action is mapped from ``[-1, 1]`` to ``[low, high]``, so the network outputs valid actions on its own.

    Args:
        log_std_min: Lower clamp of ``log σ``.
        log_std_max: Upper clamp of ``log σ``.
        low: Lower action bound (scalar or one per dimension).
        high: Upper action bound.
    """

    def __init__(
        self,
        *,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        low: float | list[float] = -1.0,
        high: float | list[float] = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.low, self.high = low, high

    def activation(self, mean: Tensor, log_std: Tensor) -> dict[str, Tensor]:
        action = affine_range(torch.tanh(mean), self.low, self.high)
        return {"action": action, "mean": action}

    def sample(
        self,
        *logits: Tensor,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> ActionSample:
        mean, log_std = logits
        action, log_prob = squashed_gaussian_sample(
            mean,
            log_std,
            low=self.low,
            high=self.high,
            deterministic=deterministic,
            generator=generator,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        )
        return ActionSample(action=action, log_prob=log_prob, entropy=-log_prob)


__all__ = ["TanhGaussianBoxActivation"]
