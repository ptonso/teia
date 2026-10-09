"""Base Runner contract.

The Runner owns the **collect↔train coupling** — when experience is produced versus
consumed, how stale it may be, and on what substrate production runs. The choice of Runner
*is* the on-policy / off-policy / async mode, exactly as the buffer (a ``StreamReshape`` node)
owns trajectory math and the env reader (a ``stream``-stage ``Reader`` node) owns the simulator.

``iter_minibatches()`` drives the env through the graph's item subgraph (``TeiaDataModule.
_item_workspace``), fills the stream-stage reshape node (if declared), and yields collated
minibatches (``TeiaDataModule._collate_fn``) to the learner, raising ``StopIteration`` once the
env-step budget is reached. ``TeiaDataModule.train_dataloader`` builds a ``RunnerContext`` (the
graph-resolved reader/reshape/ctx plus seeds and device) and delegates to the configured Runner.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import torch


@dataclass(frozen=True)
class RunnerContext:
    """Everything a Runner needs to drive one ``fit`` stage.

    Assembled by ``TeiaDataModule._runner_context`` once the live policy exists (resolved via
    ``ctx["policy"]``, from ``resolve_context("policy")``). ``stream_reader`` is the graph's
    ``stage="stream"`` ``Reader``; ``stream_reshape`` is the graph's ``stage="stream"``
    ``StreamReshape`` (the buffer), or ``None`` for the bufferless on-policy default.
    """

    dm: Any
    stream_reader: Any
    stream_reshape: Any | None
    ctx: dict[str, Any]
    total_env_steps: int
    num_envs: int
    seed: int
    device: torch.device
    generator: torch.Generator
    training: bool
    episode_log: list[tuple[float, int]] | None


class Runner(ABC):
    """Owns the collect↔train coupling; the choice of Runner is the training mode."""

    def configure(self, ctx: RunnerContext) -> None:
        """Bind the runtime context (graph reader/reshape/ctx/...). Called once per fit."""
        self._ctx = ctx

    @property
    def ctx(self) -> RunnerContext:
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            raise RuntimeError("Runner used before configure(ctx) was called.")
        return ctx

    @abstractmethod
    def iter_minibatches(self) -> Iterator[Any]:
        """Yield collated minibatches; raise ``StopIteration`` at the env-step budget."""

    def close(self) -> None:
        """Tear down background producers (threads/processes). No-op for sync runners."""
        return None

    def metrics(self) -> dict[str, float]:
        """Runner diagnostics (collection/learning throughput, producer lag)."""
        return {}
