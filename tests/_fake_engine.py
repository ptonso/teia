"""Test-local engine: environment ``i`` ends its episode after ``length + i`` steps and reports same-step.

Emits, in order: ``obs`` (``[num_envs, 2]``, both columns the step counter), ``reward`` (the first action
component, zero for an idle environment), ``terminated``, ``truncated`` and, under ``all_done``, ``valid``
(false for an environment idling since it finished). ``end`` holds the final counter of a finished environment
where ``next`` holds its reset value 0.
"""

from __future__ import annotations

import torch

from teia.base.data import Engine, EngineSpec, EngineStep


class FakeEngine(Engine):
    def __init__(self, num_envs: int = 3, length: int = 3, reset_policy: str = "per_env") -> None:
        self.num_envs, self.length, self.reset_policy = num_envs, length, reset_policy
        self.opened = 0
        self._t: torch.Tensor | None = None

    seed = 0

    def open(self, seed: int) -> None:
        if self._t is not None:
            return
        self.opened += 1
        self._t = torch.zeros(self.num_envs)
        self._idle = torch.zeros(self.num_envs, dtype=torch.bool)
        self._lens = self.length + torch.arange(self.num_envs)

    def close(self) -> None:
        self._t = None

    def spec(self) -> EngineSpec:
        outputs = [((2,), "float32", "state"), ((), "float32", "outcome"), ((), "bool", "outcome"), ((), "bool", "outcome")]
        if self.reset_policy == "all_done":
            outputs.append(((), "bool", "outcome"))
        return EngineSpec(self.num_envs, tuple(outputs), (1,), "float32", -1.0, 1.0)

    def _pack(self, obs_t, reward, terminated, truncated, valid):
        out = (obs_t.unsqueeze(1).repeat(1, 2), reward, terminated, truncated)
        return out + (valid,) if self.reset_policy == "all_done" else out

    def reset(self):
        self._t.zero_()
        self._idle.zero_()
        zeros = torch.zeros(self.num_envs)
        return self._pack(self._t.clone(), zeros, zeros.bool(), zeros.bool(), ~self._idle)

    def step(self, action: torch.Tensor) -> EngineStep:
        active = ~self._idle
        t_end = torch.where(active, self._t + 1, self._t)
        done = active & (t_end >= self._lens)
        reward = action[:, 0].float() * active
        end = self._pack(t_end.clone(), reward, done, torch.zeros_like(done), active)
        if self.reset_policy == "per_env":
            self._t = torch.where(done, torch.zeros_like(t_end), t_end)
        else:
            self._t, self._idle = t_end, self._idle | done
            if self._idle.all():
                self._t, self._idle = torch.zeros_like(t_end), torch.zeros_like(self._idle)
        nxt = self._pack(self._t.clone(), reward, done, torch.zeros_like(done), ~self._idle)
        return EngineStep(nxt, end, done)
