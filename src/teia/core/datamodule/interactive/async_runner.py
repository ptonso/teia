"""Asynchronous Runner — overlaps collection with training (the parallel collect+train mode).

A background **producer** fills the buffer while the **learner** (Lightning's main thread) drains
``buffer.sample_iter()`` continuously. This is correct *by construction* only for off-policy (replay) buffers: the
producer acts with weights that drift a few updates behind the learner, which replay already tolerates. A rollout
(on-policy) buffer under async would train on a stale, never-finalized rollout, so it is rejected; use ``SyncRunner``.

``parallel`` selects the production substrate:

* ``inline`` — one collect step per learner pull, no thread. Deterministic; same observable behavior as ``SyncRunner``
  on a replay buffer. Used for tests/debugging.
* ``thread`` — a daemon ``threading.Thread`` runs the collection loop and appends to the buffer under a lock; the
  learner drains under the same lock. The producer shares the learner's module memory, so weights are live with no
  broadcast. Engine stepping and torch inference release the GIL, so collection overlaps the learner's compute.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

from teia.core.datamodule.interactive._cycle import Cycle
from teia.core.datamodule.interactive.runner_base import Runner


class AsyncRunner(Runner):
    def __init__(self, *, parallel: str = "thread", **config: Any) -> None:
        super().__init__(**config)
        if parallel not in ("inline", "thread"):
            raise ValueError(f"AsyncRunner.parallel must be 'inline' or 'thread'; got {parallel!r}.")
        self._parallel = parallel
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._collect_error: BaseException | None = None

    def iter_minibatches(self) -> Iterator[Any]:
        ctx = self.ctx
        if ctx.buffer is None or callable(getattr(ctx.buffer, "finalize_rollout", None)):
            raise ValueError("AsyncRunner requires an off-policy buffer (replay); a rollout buffer needs SyncRunner.")
        ctx.buffer.configure(num_envs=ctx.num_envs, generator=ctx.generator)
        ctx.dm._set_training(ctx.training)
        self._cycle = Cycle(ctx, deterministic=False)
        yield from (self._inline() if self._parallel == "inline" else self._threaded())

    def _collect_once(self) -> None:
        row = self._cycle.step()
        with self._lock:
            self.ctx.buffer.add(row)

    def _inline(self) -> Iterator[Any]:
        ctx = self.ctx
        while self._cycle.env_steps < self.total_env_steps:
            self._collect_once()
            if ctx.buffer.ready():
                for minibatch in ctx.buffer.sample_iter():
                    yield ctx.dm._collate_fn(minibatch)
                ctx.buffer.on_consumed()

    def _threaded(self) -> Iterator[Any]:
        self._stop.clear()
        self._collect_error = None
        self._thread = threading.Thread(target=self._collect_loop, name="stream-collector", daemon=True)
        self._thread.start()
        try:
            while True:
                batches = self._drain_ready()
                if batches:
                    yield from batches
                    continue
                if self._collect_error is not None:
                    raise self._collect_error
                if not self._thread.is_alive():
                    batches = self._drain_ready()
                    if batches:
                        yield from batches
                        continue
                    break
                time.sleep(0)
        finally:
            self.close()

    def _drain_ready(self) -> list[Any] | None:
        buffer = self.ctx.buffer
        with self._lock:
            if not buffer.ready():
                return None
            minibatches = list(buffer.sample_iter())
            buffer.on_consumed()
        return [self.ctx.dm._collate_fn(minibatch) for minibatch in minibatches]

    def _collect_loop(self) -> None:
        try:
            while not self._stop.is_set() and self._cycle.env_steps < self.total_env_steps:
                self._collect_once()
        except BaseException as err:
            self._collect_error = err

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def drain_episodes(self) -> list[dict[str, Any]]:
        cycle = getattr(self, "_cycle", None)
        if cycle is None:
            return []
        with self._lock:
            return cycle.drain_episodes()

    def metrics(self) -> dict[str, float]:
        cycle = getattr(self, "_cycle", None)
        if cycle is None:
            return {}
        with self._lock:
            return {**cycle.drain_metrics(), **self.ctx.buffer.metrics()}
