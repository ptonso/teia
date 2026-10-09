"""Asynchronous Runner — overlaps collection with training (the parallel collect+train mode).

A background **producer** fills the stream-reshape node (the buffer) while the **learner**
(Lightning's main thread) drains ``stream_reshape.sample_iter()`` continuously. This is correct
*by construction* only for off-policy (replay) reshape nodes: the producer acts with weights
that drift a few updates behind the learner, which replay already tolerates. A rollout
(on-policy) reshape node under async would train on a stale, never-finalized rollout, so it is
rejected — use ``SyncRunner`` there.

``parallel`` selects the production substrate:

* ``inline`` — degenerate, one collect step per learner pull, no thread. Deterministic;
  same observable behavior as ``SyncRunner`` on a replay buffer. Used for tests/debugging.
* ``thread`` — a daemon ``threading.Thread`` runs the collection loop and appends to the
  reshape node under a lock; the learner drains under the same lock. The producer shares the
  learner's module memory, so weights are live with no broadcast. Env stepping and torch
  inference release the GIL, so collection genuinely overlaps the learner's compute.

A future ``subproc`` substrate (Ape-X/IMPALA-style actor processes) would not share memory
and would need explicit ``state_dict`` weight broadcast; it is out of scope here.

"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

from teia.core.datamodule.interactive._collect import CollectState
from teia.core.datamodule.interactive.runner_base import Runner


class AsyncRunner(Runner):
    """Overlap collection with training via an inline or threaded off-policy producer."""

    def __init__(self, parallel: str = "thread") -> None:
        if parallel not in ("inline", "thread"):
            raise ValueError(
                f"AsyncRunner.parallel must be 'inline' or 'thread' "
                f"(subproc is a future substrate); got {parallel!r}."
            )
        self._parallel = parallel
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._collect_error: BaseException | None = None

    def iter_minibatches(self) -> Iterator[Any]:
        ctx = self.ctx
        if ctx.stream_reshape is None or callable(getattr(ctx.stream_reshape, "finalize_rollout", None)):
            raise ValueError(
                "AsyncRunner requires an off-policy reshape node (e.g. replay): the rollout-free "
                "default and on-policy rollout buffers would train on a stale, never-finalized "
                "rollout under async. Use SyncRunner."
            )
        ctx.stream_reshape.configure(num_envs=ctx.num_envs, generator=ctx.generator)
        if self._parallel == "inline":
            yield from self._inline()
        else:
            yield from self._threaded()

    # -- inline (no thread) -------------------------------------------------------
    def _inline(self) -> Iterator[Any]:
        ctx = self.ctx
        state = CollectState(ctx, self._lock)
        while state.env_step < ctx.total_env_steps:
            state.step_once()
            if ctx.stream_reshape.ready():
                for minibatch_seeds in ctx.stream_reshape.sample_iter():
                    yield ctx.dm._collate_fn(minibatch_seeds)
                ctx.stream_reshape.on_consumed()

    # -- threaded -----------------------------------------------------------------
    def _threaded(self) -> Iterator[Any]:
        self._stop.clear()
        self._collect_error = None
        self._thread = threading.Thread(target=self._collect_loop, name="rl-collector", daemon=True)
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
                    # Producer finished; drain any final ready-cycle, then stop.
                    batches = self._drain_ready()
                    if batches:
                        yield from batches
                        continue
                    break
                time.sleep(0)  # cooperatively yield the GIL to the producer
        finally:
            self.close()

    def _drain_ready(self) -> list[Any] | None:
        """Under the lock: if the reshape node is ready, materialize one sample cycle and reset it."""
        ctx = self.ctx
        with self._lock:
            if not ctx.stream_reshape.ready():
                return None
            minibatch_seeds_list = list(ctx.stream_reshape.sample_iter())
            ctx.stream_reshape.on_consumed()
        return [ctx.dm._collate_fn(seeds) for seeds in minibatch_seeds_list]

    def _collect_loop(self) -> None:
        ctx = self.ctx
        try:
            state = CollectState(ctx, self._lock)
            while not self._stop.is_set() and state.env_step < ctx.total_env_steps:
                state.step_once()
        except BaseException as err:  # surfaced to the learner thread in _threaded
            self._collect_error = err

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None
