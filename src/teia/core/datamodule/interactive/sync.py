"""Synchronous Runner — collection and training never overlap (one thread, two phases).

The on-policy / off-policy distinction is carried by the **buffer** (a ``StreamReshape`` node), not the Runner:

* Rollout buffer (on-policy) — ``ready()`` fires after a full rollout; ``finalize_rollout()`` computes GAE from the
  value captured at the state after the last step; ``sample_iter()`` drains ``update_epochs × minibatches``;
  ``on_consumed()`` clears the buffer.
* Replay buffer (off-policy) — ``ready()`` fires after warmup/``train_frequency``; ``sample_iter()`` yields
  ``updates_per_cycle`` sampled minibatches; ``on_consumed()`` is a no-op (the buffer persists).

One environment step is one engine step plus one capture-slice forward (teia:core/datamodule.md#stream-mode).
"""

from __future__ import annotations

from typing import Any, Iterator

from teia.core.datamodule.interactive._cycle import Cycle
from teia.core.datamodule.interactive.runner_base import Runner


class SyncRunner(Runner):
    def iter_minibatches(self) -> Iterator[Any]:
        ctx = self.ctx
        buffer = ctx.buffer
        if buffer is None:
            raise ValueError("a stream-mode datamodule declares a `stage: stream` buffer node (reshape).")
        buffer.configure(num_envs=ctx.num_envs, generator=ctx.generator)
        ctx.dm._set_training(ctx.training)
        self._cycle = cycle = Cycle(ctx, deterministic=False, return_key=self.return_key, episode_id=self.episode_id)
        rollout = callable(getattr(buffer, "finalize_rollout", None))
        while cycle.env_steps < self.total_env_steps:
            buffer.add(cycle.step())
            if buffer.ready():
                if rollout:
                    buffer.finalize_rollout(last_value=cycle.atom_for(buffer.value_key))
                for minibatch in buffer.sample_iter():
                    yield ctx.dm._collate_fn(minibatch)
                buffer.on_consumed()

    def drain_episodes(self) -> list[tuple[float, int]]:
        cycle = getattr(self, "_cycle", None)
        return [] if cycle is None else cycle.drain_episodes()

    def metrics(self) -> dict[str, float]:
        cycle = getattr(self, "_cycle", None)
        return {} if cycle is None else {**cycle.drain_metrics(), **self.ctx.buffer.metrics()}
