"""``InteractiveRegime`` — couples online collection to training for an engine-sourced datamodule.

``TeiaDataModule`` delegates here when its data graph carries an engine node (stream mode). The regime builds the
``RunnerContext`` for training, runs deterministic evaluation on the already-open engine, and probes shapes for
``batch_meta()``. See teia:core/datamodule.md#stream-mode.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import torch
from torch.utils.data import IterableDataset

from teia.core.datamodule.interactive._cycle import END_PHASES, Cycle, build_row
from teia.core.datamodule.interactive.policy import FeedforwardPolicyAdapter
from teia.core.datamodule.interactive.runner_base import ACTION_ATOM, RunnerContext


class InteractiveRegime:
    def __init__(self, dm: Any) -> None:
        self._dm = dm

    def context(self, module: Any, *, training: bool) -> RunnerContext:
        dm = self._dm
        engine = dm._engine_node
        feedback = {atom: f"item.{field}" for atom, field in dm.contract.feedback.items()}
        if ACTION_ATOM not in feedback:
            raise ValueError(f"the task contract's `feedback` must return '{ACTION_ATOM}'.")
        return RunnerContext(
            dm=dm,
            engine=engine,
            buffer=dm._stream_reshape_node(),
            adapter=FeedforwardPolicyAdapter(module),
            feedback=feedback,
            episode_id=dm._runner.episode_id,
            return_key=dm._runner.return_key,
            num_envs=engine.spec().num_envs,
            device=_policy_device(module),
            generator=torch.Generator().manual_seed(int(engine.seed)),
            training=training,
        )

    def train_iterable(self, module: Any) -> IterableDataset:
        dm = self._dm
        dm._runner.configure(self.context(module, training=True))
        return _RunnerIterable(dm._runner)

    def eval_iterable(self, module: Any) -> IterableDataset:
        return _StreamEvalDataset(self, module)

    def play(self, module: Any, *, until: Callable[[Cycle], bool], deterministic: bool = True) -> Iterator[tuple[Cycle, dict[str, Any]]]:
        """Run the cycle on the open engine with frozen item nodes until ``until(cycle)``, yielding each step's row.
        The engine is reset before and after, so a training cycle in flight restarts from a fresh state."""
        dm = self._dm
        ctx = self.context(module, training=False)
        previous = dm._set_training(False)
        dm._engine_epoch += 1
        try:
            cycle = Cycle(ctx, deterministic=deterministic, episode_id=dm._runner.episode_id)
            while not until(cycle):
                yield cycle, cycle.step()
        finally:
            dm._restore_training(previous)
            dm._engine_epoch += 1
            ctx.engine.reset()

    def eval_steps(self, module: Any) -> Iterator[Any]:
        """Per-step collated batches of the deterministic cycle until ``eval_episodes`` episodes ended."""
        episodes = self._dm._runner.eval_episodes
        for _, row in self.play(module, until=lambda cycle: cycle.ended >= episodes):
            yield self._dm._collate_env_axis(row)

    def probe_batch(self) -> Any:
        """One collated step from a reset engine, a zero action and zero placeholders for the other fed-back atoms,
        run with frozen statistics. Fields no node produces (buffer outputs) are absent."""
        dm = self._dm
        engine = dm._engine_node
        engine.open(engine.seed)
        spec = engine.spec()
        n = spec.num_envs
        keys = dm._engine_rec.out_key
        values = engine.reset()
        feedback = {}
        for atom, field in dm.contract.feedback.items():
            if atom == ACTION_ATOM:
                dtype = torch.int64 if spec.num_actions is not None else getattr(torch, spec.action_dtype)
                shape = () if spec.num_actions is not None else spec.action_shape
                feedback[f"item.{field}"] = torch.zeros((n, *shape), dtype=dtype)
            else:
                feedback[f"item.{field}"] = torch.zeros(n)
        previous = dm._set_training(False)
        try:
            state = {k: v for k, v, (_, _, role) in zip(keys, values, spec.outputs) if role == "state"}
            cur = dm._stream_pass(state, frozenset({"preprocess"}))
            end = dm._stream_pass({**dict(zip(keys, values)), **feedback}, END_PHASES)
            return dm._collate_env_axis(build_row(cur, end))
        finally:
            dm._restore_training(previous)


class _RunnerIterable(IterableDataset):
    """Thin bridge from a configured ``Runner`` to Lightning's training loop."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def __iter__(self) -> Iterator[Any]:
        try:
            yield from self._runner.iter_minibatches()
        finally:
            self._runner.close()


class _StreamEvalDataset(IterableDataset):
    """Capture-only eval iterable: yields per-step collated batches carrying the episode ids."""

    def __init__(self, regime: InteractiveRegime, module: Any) -> None:
        self._regime, self._module = regime, module

    def __iter__(self) -> Iterator[Any]:
        yield from self._regime.eval_steps(self._module)


def _policy_device(module: Any) -> torch.device:
    dev = getattr(module, "device", None)
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, str):
        return torch.device(dev)
    return torch.device("cpu")
