"""
Gaussian Box Action Activation.

Source:
  - title: "Proximal Policy Optimization Algorithms"
    url: "https://arxiv.org/abs/1707.06347"
    year: 2017
  - title: "Upstream implementation - CleanRL ppo_continuous_action.py"
    url: "https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/ppo_continuous_action.py"
    year: null

Description:
  Converts a Gaussian mean and log standard deviation into continuous box actions and log probabilities. The activation samples from a diagonal Gaussian and emits the raw sample (for the log-probability the buffer keeps) next to a copy clamped to the action bounds (for the simulator).

Adaptations:
  - Takes the mean and log standard deviation as two inputs (the log standard deviation may be a state-independent parameter).
  - Emits `env_action`, the sample clamped to `[low, high]`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import Tensor

from teia.base.net import ActionActivation, ActionSample
from teia.node.net._activation_utils import decode_policy
from teia.node.net._gaussian_policy_utils import LOG_STD_MAX, LOG_STD_MIN, clamp_log_std


class GaussianSample(NamedTuple):
    """``ActionSample`` plus the bound-clamped copy for the engine."""

    action: Tensor
    log_prob: Tensor | None
    entropy: Tensor | None
    env_action: Tensor


class GaussianBoxActivation(ActionActivation):
    """Diagonal Gaussian policy over a Box action space.

        a = μ + σ ε            sample;  μ when deterministic and in the traced route
        env_action = clamp(a, low, high)

    ``action`` is the raw sample (its log-probability is exact); ``env_action`` is what the engine receives.

    Args:
        log_std_min: Lower clamp of ``log σ``.
        log_std_max: Upper clamp of ``log σ``.
        low: Lower action bound (scalar or one per dimension); null leaves ``env_action`` unclamped.
        high: Upper action bound.
    """

    @staticmethod
    def kernel(activated: Mapping[str, Any], ctx: Mapping[str, Any]) -> list[dict]:
        return decode_policy(activated, ctx)

    def __init__(
        self,
        *,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        low: float | list[float] | None = None,
        high: float | list[float] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.low, self.high = low, high

    def _clamp(self, action: Tensor) -> Tensor:
        if self.low is None:
            return action
        low, high = (torch.as_tensor(b, dtype=action.dtype, device=action.device) for b in (self.low, self.high))
        return torch.minimum(torch.maximum(action, low), high)

    # -- prediction routes --------------------------------------------------------
    def activation(self, mean: Tensor, log_std: Tensor) -> dict[str, Tensor]:
        """Deterministic, ONNX-traceable route: the mean action and its log-probability."""
        std = clamp_log_std(mean, log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max).exp()
        log_prob = torch.distributions.Normal(mean, std).log_prob(mean).sum(-1)
        return {"action": mean, "env_action": self._clamp(mean), "mean": mean, "log_prob": log_prob}

    def sample(
        self,
        *logits: Tensor,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> GaussianSample:
        """Stochastic collection/eval route (not ONNX-traced)."""
        mean, log_std = logits
        std = clamp_log_std(mean, log_std, log_std_min=self.log_std_min, log_std_max=self.log_std_max).exp()

        # a = μ + σ ε, or μ when deterministic
        if deterministic:
            raw = mean
        else:
            noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
            raw = mean + std * noise.to(mean.device)

        # log π(a|s) and entropy, summed over action dims
        dist = torch.distributions.Normal(mean, std)
        return GaussianSample(raw, dist.log_prob(raw).sum(-1), dist.entropy().sum(-1), self._clamp(raw))
