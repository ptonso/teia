"""Off-policy buffers: uniform replay with n-step returns, prioritized replay and a sequence buffer.

Storage is a ring of ``capacity // num_envs`` steps, each holding one transition per env. ``ready()`` fires once
``warmup_steps`` transitions were added and ``train_frequency`` more arrived since the last ``on_consumed()``, which
resets the cadence and keeps the data. The buffer reads the keys bound by its mapping-form ``in`` (``reward``,
``terminated``, ``truncated`` and optionally ``valid``, the engine's validity mask under ``all_done``) and writes the
keys of its positional ``out``. Successor features are the ``item.next_<K>`` keys the runner stores in each row.
"""

from __future__ import annotations

from typing import Any, Iterator

import torch

from teia.base.data.nodes import StreamReshape
from teia.core.datamodule.interactive._roles import bind_roles
from teia.core.datamodule.interactive._tensor_ops import step_rows


class ReplayBuffer(StreamReshape):
    """Uniform replay. A sample is the transition at step ``t`` with its n-step return: the reward sums
    ``gamma^k * r`` over the window, which ends early at an episode end or at the newest stored step, and
    the ``n_step_discount`` output is ``gamma^len`` (the factor the bootstrap term needs). ``terminated`` is the flag of
    the last step in the window, so a truncation still bootstraps. Outputs, in order: ``weight``, ``index``,
    ``n_step_discount``."""

    IN_ROLES = ("reward", "terminated", "truncated")
    OUT_ROLES = ("weight", "index", "n_step_discount")

    def __init__(
        self,
        capacity: int = 1_000_000,
        warmup_steps: int = 10_000,
        train_frequency: int = 1,
        updates_per_cycle: int = 1,
        minibatch_size: int = 64,
        n_step: int = 1,
        gamma: float = 0.99,
        **_: Any,
    ) -> None:
        self.capacity = capacity
        self.warmup_steps = warmup_steps
        self.train_frequency = train_frequency
        self.updates_per_cycle = updates_per_cycle
        self.minibatch_size = minibatch_size
        self.n_step = n_step
        self.gamma = gamma

    def configure(self, num_envs: int, generator: torch.Generator | None = None, **_: Any) -> None:
        self._roles, self._out = bind_roles(self, required=self.IN_ROLES, optional=("valid",), outs=self.OUT_ROLES)
        self._num_envs = num_envs
        self._generator = generator or torch.Generator()
        self._max_steps = max(self.capacity // num_envs, 1)
        self._store: dict[str, torch.Tensor] = {}
        self._size = 0
        self._pos = 0
        self._added = 0
        self._cadence = 0

    def add(self, item: Any) -> None:
        for key, row in step_rows(item, self._num_envs).items():
            if key not in self._store:
                self._store[key] = torch.zeros((self._max_steps, *row.shape), dtype=row.dtype)
            self._store[key][self._pos] = row
        self._on_write(self._pos)
        self._pos = (self._pos + 1) % self._max_steps
        self._size = min(self._size + 1, self._max_steps)
        self._added += self._num_envs
        self._cadence += self._num_envs

    def _on_write(self, pos: int) -> None:
        """Hook for subclasses that keep per-slot state."""

    def ready(self) -> bool:
        return self._size > 0 and self._added >= self.warmup_steps and self._cadence >= self.train_frequency

    def on_consumed(self) -> None:
        self._cadence = 0

    def metrics(self) -> dict[str, float]:
        return {"buffer/size": float(self._size * self._num_envs), "buffer/added": float(self._added)}

    def sample_iter(self) -> Iterator[list[dict[str, torch.Tensor]]]:
        for _ in range(self.updates_per_cycle):
            yield self._batch(self._draw(self.minibatch_size))

    def _valid(self) -> torch.Tensor:
        """Flat ``step * num_envs + env`` indices that may start a sample."""
        keep = torch.ones(self._size, self._num_envs, dtype=torch.bool)
        if "valid" in self._roles:
            keep = self._store[self._roles["valid"]][: self._size] > 0
        return keep.reshape(-1).nonzero().reshape(-1)

    def _draw(self, count: int) -> torch.Tensor:
        valid = self._valid()
        return valid[torch.randint(len(valid), (count,), generator=self._generator)]

    def _batch(self, flat: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [self._assemble(*divmod(int(i), self._num_envs)) for i in flat]

    def _assemble(self, step: int, env: int) -> dict[str, torch.Tensor]:
        r, term, trunc = (self._roles[role] for role in self.IN_ROLES)
        ahead = (self._pos - 1 - step) % self._max_steps
        reward, taken = 0.0, 0
        while True:
            last = (step + taken) % self._max_steps
            reward += self.gamma**taken * float(self._store[r][last, env])
            taken += 1
            ended = bool(self._store[term][last, env] | self._store[trunc][last, env])
            if ended or taken >= self.n_step or taken > ahead:
                break
        record = {key: tensor[step, env].clone() for key, tensor in self._store.items()}
        for key, tensor in self._store.items():
            if key.startswith("item.next_") or key in (term, trunc):
                record[key] = tensor[last, env].clone()
        record[r] = torch.tensor(reward, dtype=self._store[r].dtype)
        record[self._out["n_step_discount"]] = torch.tensor(self.gamma**taken)
        record[self._out["weight"]] = torch.ones(())
        record[self._out["index"]] = torch.tensor(step * self._num_envs + env)
        return record


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay: draws with probability ``priority^alpha`` and corrects with importance weights
    ``(N * P)^-beta`` normalized by the batch maximum. New steps get the running maximum priority, and
    ``update_priorities(index, priority)`` sets the priority of ``item.index`` rows to ``|priority| + eps``."""

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-6, **kwargs: Any) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        super().__init__(**kwargs)

    def configure(self, num_envs: int, generator: torch.Generator | None = None, **kwargs: Any) -> None:
        super().configure(num_envs, generator, **kwargs)
        self._priority = torch.zeros(self._max_steps, num_envs)
        self._max_priority = 1.0

    def _on_write(self, pos: int) -> None:
        self._priority[pos] = self._max_priority

    def _draw(self, count: int) -> torch.Tensor:
        valid = self._valid()
        weights = self._priority.reshape(-1)[valid] ** self.alpha
        probs = weights / weights.sum()
        chosen = torch.multinomial(probs, count, replacement=True, generator=self._generator)
        self._draw_weights = (len(valid) * probs[chosen]) ** -self.beta
        return valid[chosen]

    def _batch(self, flat: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        records = super()._batch(flat)
        weights = self._draw_weights / self._draw_weights.max()
        for record, weight in zip(records, weights, strict=True):
            record[self._out["weight"]] = weight
        return records

    def update_priorities(self, index: torch.Tensor, priority: torch.Tensor) -> None:
        values = priority.detach().abs().cpu() + self.eps
        self._priority.reshape(-1)[index.cpu()] = values
        self._max_priority = max(self._max_priority, float(values.max()))


class SequenceBuffer(ReplayBuffer):
    """Samples windows of ``sequence_length`` consecutive steps of one env, ``[T, ...]`` per field. A window ends after
    the first episode end or at the newest stored step, and is zero-padded. Its single output is the padding mask
    (1 for a real step that the optional ``valid`` role admits)."""

    OUT_ROLES = ("pad_mask",)

    def __init__(self, sequence_length: int = 16, **kwargs: Any) -> None:
        self.sequence_length = sequence_length
        super().__init__(**kwargs)

    def _batch(self, flat: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [self._window(*divmod(int(i), self._num_envs)) for i in flat]

    def _window(self, step: int, env: int) -> dict[str, torch.Tensor]:
        ahead = (self._pos - 1 - step) % self._max_steps
        term, trunc = self._roles["terminated"], self._roles["truncated"]
        steps = [(step + k) % self._max_steps for k in range(min(ahead, self.sequence_length - 1) + 1)]
        for length, slot in enumerate(steps, start=1):
            if bool(self._store[term][slot, env] | self._store[trunc][slot, env]):
                steps = steps[:length]
                break
        window = {}
        for key, tensor in self._store.items():
            padded = torch.zeros((self.sequence_length, *tensor.shape[2:]), dtype=tensor.dtype)
            padded[: len(steps)] = tensor[steps, env]
            window[key] = padded
        pad = torch.zeros(self.sequence_length)
        pad[: len(steps)] = self._store[self._roles["valid"]][steps, env].float() if "valid" in self._roles else 1.0
        window[self._out["pad_mask"]] = pad
        return window
