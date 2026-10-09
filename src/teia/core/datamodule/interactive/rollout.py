"""``RolloutBuffer`` — the on-policy buffer: collect a fixed number of steps, compute GAE, yield shuffled minibatches, discard.

Advantages use ``delta_t = r_t + gamma * (1 - terminated_t) * V_{t+1} - V_t`` and cut the trace at every episode end
(``terminated`` or ``truncated``). A truncated step still bootstraps: under next-step autoreset the step after it is the
masked final-observation step, so ``V_{t+1}`` is the value of the true final observation. Masked steps (``item.mask == 0``)
are dropped from the training batch.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from teia.base.data.nodes import StreamReshape
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

    def configure(self, num_envs: int, generator: torch.Generator | None = None, **_: Any) -> None:
        self._num_envs = num_envs
        self._generator = generator or torch.Generator()
        self.on_consumed()

    def add(self, item: Any) -> None:
        self._steps.append(step_rows(item, self._num_envs))

    def ready(self) -> bool:
        return len(self._steps) >= self.rollout_steps

    def finalize_rollout(self, last_value: torch.Tensor, last_done: torch.Tensor) -> None:
        """GAE over the collected steps. ``last_value`` is ``V`` of the observation after the last step; ``last_done`` is
        unused because each step already carries its own ``terminated`` and ``truncated`` flags."""
        del last_done
        rollout = {key: torch.stack([step[key] for step in self._steps]) for key in self._steps[0]}
        steps = len(self._steps)

        def per_env(key: str) -> torch.Tensor:
            return rollout[key].float().reshape(steps, self._num_envs)

        reward, value, terminated = per_env("item.reward"), per_env("item.value"), per_env("item.terminated")
        done = torch.maximum(terminated, per_env("item.truncated"))
        next_value = torch.cat([value[1:], last_value.reshape(1, self._num_envs).float()])
        delta = reward + self.gamma * (1.0 - terminated) * next_value - value
        advantage = torch.zeros_like(delta)
        trace = torch.zeros_like(delta[0])
        for t in reversed(range(delta.shape[0])):
            trace = delta[t] + self.gamma * self.gae_lambda * (1.0 - done[t]) * trace
            advantage[t] = trace
        rollout["item.return_"] = advantage + value
        keep = per_env("item.mask") > 0 if "item.mask" in rollout else torch.ones_like(reward, dtype=torch.bool)
        if self.normalize_advantage and keep.sum() > 1:
            kept = advantage[keep]
            advantage = (advantage - kept.mean()) / (kept.std() + 1e-8)
        rollout["item.advantage"] = advantage
        self._samples = {key: tensor[keep] for key, tensor in rollout.items()}

    def sample_iter(self) -> Iterator[list[dict[str, torch.Tensor]]]:
        if self._samples is None:
            raise RuntimeError("RolloutBuffer.sample_iter() before finalize_rollout().")
        size = len(self._samples["item.advantage"])
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
