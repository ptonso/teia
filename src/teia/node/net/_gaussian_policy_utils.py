"""
Gaussian Policy Utilities.

Source:
  - title: "Proximal Policy Optimization Algorithms"
    url: "https://arxiv.org/abs/1707.06347"
    year: 2017
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018

Description:
  Builds diagonal Gaussian distributions, log probabilities, entropies, and tanh-squashed samples from a mean and a log standard deviation. The module centralizes continuous-control policy math shared by activations, sampler nodes, and losses.

Adaptations:
  - Takes mean and log standard deviation as two tensors (the log standard deviation may be a state-independent parameter) with configurable clipping bounds.

"""

from __future__ import annotations

import torch
from torch import Tensor

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def clamp_log_std(mean: Tensor, log_std: Tensor, *, log_std_min: float = LOG_STD_MIN, log_std_max: float = LOG_STD_MAX) -> Tensor:
    """``log σ`` clamped to the bounds and broadcast to the shape of ``mean`` (a state-independent parameter has no batch axis)."""
    return log_std.clamp(log_std_min, log_std_max).expand_as(mean)


def gaussian_distribution(mean: Tensor, log_std: Tensor, **clamp: float) -> torch.distributions.Normal:
    """The diagonal Gaussian ``Normal(mean, exp(log_std))``."""
    return torch.distributions.Normal(mean, clamp_log_std(mean, log_std, **clamp).exp())


def gaussian_log_prob(mean: Tensor, log_std: Tensor, action: Tensor, **clamp: float) -> Tensor:
    """Return the summed per-dimension log-probability of ``action`` under the Gaussian policy."""
    return gaussian_distribution(mean, log_std, **clamp).log_prob(action).sum(-1)


def gaussian_entropy(mean: Tensor, log_std: Tensor, **clamp: float) -> Tensor:
    """Return the summed per-dimension entropy of the Gaussian policy."""
    return gaussian_distribution(mean, log_std, **clamp).entropy().sum(-1)


def affine_range(action: Tensor, low: float | list[float], high: float | list[float]) -> Tensor:
    """Map an action in ``[-1, 1]`` to ``[low, high]`` (scalars or one bound per action dimension)."""
    low, high = torch.as_tensor(low, dtype=action.dtype, device=action.device), torch.as_tensor(high, dtype=action.dtype, device=action.device)
    return low + (action + 1.0) * 0.5 * (high - low)


def squashed_gaussian_sample(
    mean: Tensor,
    log_std: Tensor,
    *,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
    log_std_min: float = LOG_STD_MIN,
    log_std_max: float = LOG_STD_MAX,
    low: float | list[float] = -1.0,
    high: float | list[float] = 1.0,
) -> tuple[Tensor, Tensor]:
    """Sample a tanh-squashed Gaussian action mapped to ``[low, high]`` and return ``(action, log_prob)``.

    ``log_prob`` stays on the ``[-1, 1]`` variable: the affine map shifts it by a constant, which leaves gradients and
    the target entropy unchanged.
    """
    std = clamp_log_std(mean, log_std, log_std_min=log_std_min, log_std_max=log_std_max).exp()
    if deterministic:
        pre_tanh = mean
    else:
        # randn_like (not randn(mean.shape, ...)) and validate_args=False below: both are required
        # for torch.export to trace this under a dynamic batch dim (SymIntArrayRef / data-dependent
        # guard errors otherwise). The ONNX registry has no symbolic function for the `generator`
        # overload, so only pass generator= when one is actually given (selects the `.default`
        # overload); a passed generator is CPU-only (torch.Generator() default) while mean may be on
        # GPU, so generate on the generator's device then move, same as the pre-fix behavior.
        if generator is not None:
            noise = torch.randn_like(mean, generator=generator, device=generator.device).to(mean.device)
        else:
            noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
    squashed = torch.tanh(pre_tanh)
    base = torch.distributions.Normal(mean, std, validate_args=False).log_prob(pre_tanh).sum(-1)
    correction = torch.log(1.0 - squashed.pow(2) + 1e-6).sum(-1)
    return affine_range(squashed, low, high), base - correction
