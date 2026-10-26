"""Test-local engine: environment ``i`` ends its episode after ``length + i`` steps and never resets itself.

Signals: ``obs`` (``[num_envs, 2]``, both columns the step counter), ``reward`` (the first action component),
``terminated`` (the counter reached the environment's length) and ``truncated`` (never). ``reset(mask)`` zeroes the
counters of the masked environments, so a finished environment's final counter is visible until the runner resets it.
"""

from __future__ import annotations

import torch

from teia.base.data import Engine, EngineSpec

_SIGNALS = {"obs": ((2,), "float32", "state"), "reward": ((), "float32", "outcome"), "terminated": ((), "bool", "outcome"), "truncated": ((), "bool", "outcome")}


class FakeEngine(Engine):
    def __init__(self, num_envs: int = 3, length: int = 3) -> None:
        self.num_envs, self.length = num_envs, length
        self.opened = 0
        self._t: torch.Tensor | None = None

    seed = 0

    def open(self, seed: int) -> None:
        unknown = [s for s in self.signals if s not in _SIGNALS]
        if unknown:
            raise ValueError(f"unknown signal(s) {unknown}")
        if self._t is not None:
            return
        self.opened += 1
        self._t = torch.zeros(self.num_envs)
        self._lens = self.length + torch.arange(self.num_envs)

    def close(self) -> None:
        self._t = None

    def spec(self) -> EngineSpec:
        return EngineSpec(self.num_envs, tuple(_SIGNALS[s] for s in self.signals), (1,), "float32", -1.0, 1.0)

    def _signals(self, reward: torch.Tensor) -> tuple[torch.Tensor, ...]:
        done = self._t >= self._lens
        values = {"obs": self._t.clone().unsqueeze(1).repeat(1, 2), "reward": reward, "terminated": done, "truncated": torch.zeros_like(done)}
        return tuple(values[s] for s in self.signals)

    def reset(self, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        self.open(self.seed)
        self._t = torch.zeros_like(self._t) if mask is None else torch.where(mask, torch.zeros_like(self._t), self._t)
        return self._signals(torch.zeros(self.num_envs))

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._t = self._t + 1
        return self._signals(action[:, 0].float())
