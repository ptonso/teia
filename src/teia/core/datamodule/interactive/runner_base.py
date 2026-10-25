"""Base Runner contract.

The Runner owns the **collect↔train coupling**: when experience is produced versus consumed, how stale it may be,
and on what substrate production runs. The choice of Runner *is* the on-policy / off-policy / async mode, exactly as
the buffer (a ``StreamReshape`` node) owns trajectory math and the engine (a ``stream``-stage ``Engine`` node) owns
the simulator.

``iter_minibatches()`` drives the cycle of teia:core/datamodule.md#stream-mode, fills the buffer and yields collated
minibatches to the learner. ``TeiaDataModule.train_dataloader`` builds a ``RunnerContext`` and delegates here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from teia.base.interactive.policy import PolicyAdapter

#: The captured atom the runner sends to the engine; a task contract with a ``feedback`` map captures it.
ACTION_ATOM = "policy.action"


@dataclass(frozen=True)
class RunnerContext:
    """Everything a Runner needs to drive one stage. ``feedback`` maps each fed-back capture atom to the
    ``item.<field>`` key it is written under."""

    dm: Any
    engine: Any
    buffer: Any | None
    adapter: PolicyAdapter
    feedback: dict[str, str]
    episode_id: str | None
    return_key: str | None
    num_envs: int
    device: torch.device
    generator: torch.Generator
    training: bool


class Runner(ABC):
    """Owns the collect↔train coupling. Config: ``total_env_steps`` (the training budget), ``eval_episodes``,
    ``episode_id`` (item key the runner writes episode ids under) and ``return_key`` (item key whose per-step
    values sum into the episode return reported by ``metrics()``)."""

    def __init__(
        self, *, total_env_steps: int, eval_episodes: int = 10, episode_id: str | None = None, return_key: str | None = None
    ) -> None:
        self.total_env_steps = int(total_env_steps)
        self.eval_episodes = int(eval_episodes)
        self.episode_id = episode_id
        self.return_key = return_key

    def configure(self, ctx: RunnerContext) -> None:
        """Bind the runtime context. Called once per fit."""
        self._ctx = ctx

    @property
    def ctx(self) -> RunnerContext:
        ctx = getattr(self, "_ctx", None)
        if ctx is None:
            raise RuntimeError("Runner used before configure(ctx) was called.")
        return ctx

    @abstractmethod
    def iter_minibatches(self) -> Iterator[Any]:
        """Yield collated minibatches until ``total_env_steps`` environment steps were collected."""

    def close(self) -> None:
        """Tear down background producers (threads/processes). No-op for sync runners."""
        return None

    def metrics(self) -> dict[str, float]:
        """Runner diagnostics: episode statistics since the last call and the buffer's own metrics."""
        return {}

    def drain_episodes(self) -> list[tuple[float, int]]:
        """``(return, length)`` of the episodes that ended since the last call (return is 0 without a ``return_key``)."""
        return []
