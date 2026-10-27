"""
Epsilon-Greedy Q Activation.

Source:
  - title: "Human-level control through deep reinforcement learning"
    url: "https://www.nature.com/articles/nature14236"
    year: 2015
  - title: "Playing Atari with Deep Reinforcement Learning"
    url: "https://arxiv.org/abs/1312.5602"
    year: 2013

Description:
  Selects discrete actions from Q-values using argmax at evaluation time and epsilon-greedy exploration during collection. The epsilon schedule controls the probability of replacing the greedy action with a random one, and counts environment steps.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from teia.base.net import ActionActivation, ActionSample


class QArgmaxActivation(ActionActivation):
    """Greedy action over Q-values, ε-greedy when sampling.

        a = argmax_a Q(s, a)        with probability 1 - ε, else uniform

    Args:
        epsilon: ``ε``: a scalar, a schedule, or a mapping for ``LinearSchedule`` (``start``, ``end``, ``duration`` in environment steps) or ``ConstantSchedule``.
    """

    def __init__(self, *, epsilon: Any = None, **_: Any) -> None:
        super().__init__()
        self._schedule = _build_schedule(epsilon)
        self._explore_step = 0

    # -- exploration step ----------------------------------------------------------
    @property
    def epsilon(self) -> float:
        return float(self._schedule.value(self._explore_step))

    def reset_exploration(self) -> None:
        self._explore_step = 0

    # -- prediction routes --------------------------------------------------------
    def activation(self, q: Tensor) -> dict[str, Tensor]:
        """Deterministic, ONNX-traceable route: the greedy action index."""
        return {"action": q.argmax(dim=-1)}

    def sample(
        self,
        *logits: Tensor,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> ActionSample:
        """Greedy (eval) or epsilon-greedy (collection) discrete action selection."""
        q = logits[0]
        greedy = q.argmax(dim=-1)
        if deterministic:
            return ActionSample(action=greedy)

        epsilon = self.epsilon
        self._explore_step += greedy.numel()
        num_actions = q.shape[-1]
        batch = greedy.shape
        # Exploration noise is drawn on the (CPU) seeded generator for reproducibility, then
        # moved to the action device — q/greedy live on the policy device during GPU collection.
        explore = (torch.rand(batch, generator=generator) < epsilon).to(greedy.device)
        random_action = torch.randint(0, num_actions, batch, generator=generator).to(greedy.device)
        action = torch.where(explore, random_action, greedy)
        return ActionSample(action=action)


def _build_schedule(epsilon: Any) -> Any:
    """Resolve the ``epsilon`` config (schedule instance / mapping / scalar / None)."""
    if epsilon is None:
        return ConstantSchedule(value=0.0)
    if hasattr(epsilon, "value") and callable(getattr(epsilon, "value")):
        return epsilon
    if isinstance(epsilon, (int, float)):
        return ConstantSchedule(value=float(epsilon))
    if isinstance(epsilon, (dict, DictConfig)):
        cfg = dict(OmegaConf.to_container(OmegaConf.create(epsilon), resolve=True))
        target = cfg.pop("_target_", None)
        if target and target.endswith("ConstantSchedule"):
            return ConstantSchedule(**cfg)
        return LinearSchedule(**cfg)
    raise TypeError(f"q_argmax epsilon must be a schedule, mapping, or scalar, got {type(epsilon).__name__}.")


class LinearSchedule:
    """Linear interpolation from ``start`` to ``end`` over ``duration`` steps, then flat."""

    def __init__(self, *, start: float = 1.0, end: float = 0.05, duration: int = 10_000, **_: Any) -> None:
        self.start = float(start)
        self.end = float(end)
        self.duration = int(duration)

    def value(self, step: int) -> float:
        if self.duration <= 0 or step >= self.duration:
            return self.end
        if step <= 0:
            return self.start
        frac = step / self.duration
        return self.start + frac * (self.end - self.start)


class ConstantSchedule:
    """A constant scalar schedule."""

    def __init__(self, *, value: float = 0.0, **_: Any) -> None:
        self._value = float(value)

    def value(self, step: int) -> float:
        return self._value


__all__ = ["QArgmaxActivation"]
