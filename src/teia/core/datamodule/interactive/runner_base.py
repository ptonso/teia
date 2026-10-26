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

@dataclass(frozen=True)
class RunnerContext:
    """Everything a Runner needs to drive one stage. ``feedback`` maps each fed-back capture atom to the
    ``item.<field>`` key it is written under."""

    dm: Any
    engine: Any
    buffer: Any | None
    adapter: PolicyAdapter
    feedback: dict[str, str]
    engine_atom: str
    boundary: list[str]
    reset_policy: str
    valid: str | None
    episode_id: str | None
    episode_sums: list[str]
    num_envs: int
    device: torch.device
    generator: torch.Generator
    training: bool


class Runner(ABC):
    """Owns the collect↔train coupling. Config: ``total_env_steps`` (the training budget), ``eval_episodes``,
    ``boundary`` (item keys whose OR ends an episode), ``reset_policy`` (``per_env`` | ``all_done``), ``valid``
    (item key the runner writes the live-environment mask under, required by ``all_done``), ``episode_id`` (item key
    the runner writes episode ids under) and ``episode_sums`` (item keys whose per-step values sum into each ended
    episode's record)."""

    def __init__(
        self,
        *,
        total_env_steps: int,
        boundary: list[str],
        eval_episodes: int = 10,
        reset_policy: str = "per_env",
        valid: str | None = None,
        episode_id: str | None = None,
        episode_sums: list[str] = (),
    ) -> None:
        if reset_policy not in ("per_env", "all_done"):
            raise ValueError(f"runner.reset_policy must be per_env or all_done; got {reset_policy!r}")
        if reset_policy == "all_done" and valid is None:
            raise ValueError("runner.reset_policy all_done needs `valid`: the item key the live mask is written under.")
        if not boundary:
            raise ValueError("runner.boundary lists the item keys whose OR ends an episode.")
        self.total_env_steps = int(total_env_steps)
        self.eval_episodes = int(eval_episodes)
        self.boundary, self.reset_policy, self.valid = [str(k) for k in boundary], reset_policy, valid
        self.episode_id = episode_id
        self.episode_sums = [str(k) for k in episode_sums]

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

    def drain_episodes(self) -> list[dict[str, Any]]:
        """Records of the episodes that ended since the last call: ``{"sums": {key: float}, "length": int, "ended_by": int}``."""
        return []
