"""``RolloutBuffer`` — the on-policy buffer: collect a fixed number of steps, compute GAE, yield shuffled minibatches, discard.

Advantages use ``delta_t = r_t + gamma * (1 - terminated_t) * V_{t+1} - V_t`` and cut the trace at every episode end
(``terminated`` or ``truncated``). Engines reset in the same step, so ``V_{t+1}`` after an episode end is the value of
the reset state: a time-limit truncation bootstraps from it, a small bias that the cut trace bounds. Rows the optional
``valid`` role marks invalid are dropped from the training batch.

The buffer reads the keys bound by its mapping-form ``in`` (``reward``, ``terminated``, ``truncated``, ``value`` and
optionally ``valid``) and writes ``return_`` then ``advantage`` through its positional ``out``.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from teia.base.data.nodes import StreamReshape
from teia.core.datamodule.interactive._roles import bind_roles
from teia.core.datamodule.interactive._tensor_ops import step_rows


class RolloutBuffer(StreamReshape):
    def __init__(
        self,
        rollout_steps: int = 2048,
        minibatch_size: int = 64,
        update_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        **_: Any,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self._num_envs = 1
        self._generator = torch.Generator()
        self._steps: list[dict[str, torch.Tensor]] = []
        self._samples: dict[str, torch.Tensor] | None = None

    @property
    def value_key(self) -> str:
        return self.in_roles["value"]

    def configure(self, num_envs: int, generator: torch.Generator | None = None, **_: Any) -> None:
        self._roles, self._out = bind_roles(
            self, required=("reward", "terminated", "truncated", "value"), optional=("valid",), outs=("return_", "advantage")
        )
        self._num_envs = num_envs
        self._generator = generator or torch.Generator()
        self.on_consumed()

    def add(self, item: Any) -> None:
        self._steps.append(step_rows(item, self._num_envs))

    def ready(self) -> bool:
        return len(self._steps) >= self.rollout_steps

    def finalize_rollout(self, last_value: torch.Tensor) -> None:
        """GAE over the collected steps. ``last_value`` is ``V`` of the state after the last step; each step carries
        its own ``terminated`` and ``truncated`` flags."""
        rollout = {key: torch.stack([step[key] for step in self._steps]) for key in self._steps[0]}
        steps = len(self._steps)

        def per_env(key: str) -> torch.Tensor:
            return rollout[key].float().reshape(steps, self._num_envs)

        reward, value, terminated = per_env(self._roles["reward"]), per_env(self._roles["value"]), per_env(self._roles["terminated"])
        done = torch.maximum(terminated, per_env(self._roles["truncated"]))
        next_value = torch.cat([value[1:], last_value.reshape(1, self._num_envs).float()])
        delta = reward + self.gamma * (1.0 - terminated) * next_value - value
        advantage = torch.zeros_like(delta)
        trace = torch.zeros_like(delta[0])
        for t in reversed(range(delta.shape[0])):
            trace = delta[t] + self.gamma * self.gae_lambda * (1.0 - done[t]) * trace
            advantage[t] = trace
        rollout[self._out["return_"]] = advantage + value
        keep = per_env(self._roles["valid"]) > 0 if "valid" in self._roles else torch.ones_like(reward, dtype=torch.bool)
        if self.normalize_advantage and keep.sum() > 1:
            kept = advantage[keep]
            advantage = (advantage - kept.mean()) / (kept.std() + 1e-8)
        rollout[self._out["advantage"]] = advantage
        self._samples = {key: tensor[keep] for key, tensor in rollout.items()}

    def sample_iter(self) -> Iterator[list[dict[str, torch.Tensor]]]:
        if self._samples is None:
            raise RuntimeError("RolloutBuffer.sample_iter() before finalize_rollout().")
        size = len(self._samples[self._out["advantage"]])
        for _ in range(self.update_epochs):
            order = torch.randperm(size, generator=self._generator)
            for start in range(0, size, self.minibatch_size):
                chosen = order[start : start + self.minibatch_size].tolist()
                yield [{key: tensor[i] for key, tensor in self._samples.items()} for i in chosen]

    def metrics(self) -> dict[str, float]:
        return {"buffer/size": float(len(self._steps) * self._num_envs)}

    def on_consumed(self) -> None:
        self._steps = []
        self._samples = None
