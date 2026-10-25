"""The stream-mode cycle (teia:core/datamodule.md#stream-mode), shared by the training runners and eval.

One ``step()`` advances every environment once: the engine steps with the action captured at the current state, two
phase-filtered item passes run (``end`` for the transition, ``next`` for the state to act from), the netmodule's capture
slice acts, and the finished row is returned. Core names no field: the engine's ``out`` keys, the task contract's
``feedback`` map and the configured runner keys are the only vocabulary.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.core.datamodule.interactive._tensor_ops import detach_cpu
from teia.core.datamodule.interactive.runner_base import ACTION_ATOM, RunnerContext

END_PHASES = frozenset({"preprocess", "target"})


def successor(key: str) -> str:
    return key.replace("item.", "item.next_", 1)


def build_row(cur: dict[str, Any], end: dict[str, Any]) -> dict[str, Any]:
    """One stored row: the state-side items of the acting step, their ``item.next_<K>`` successors from the end pass,
    and every end-pass item the state side lacks (outcomes, targets, fed-back atoms)."""
    row = {**cur, **{successor(k): end[k] for k in cur if k in end}}
    row.update({k: v for k, v in end.items() if k not in cur})
    return row


def _to(ws: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ws.items()}


class Cycle:
    def __init__(self, ctx: RunnerContext, *, deterministic: bool, return_key: str | None = None, episode_id: str | None = None) -> None:
        self.ctx, self.deterministic, self.return_key, self.episode_id = ctx, deterministic, return_key, episode_id
        self._keys = ctx.dm._engine_rec.out_key
        self._state = [i for i, (_, _, role) in enumerate(ctx.engine.spec().outputs) if role == "state"]
        self._act_phases = frozenset({"preprocess"} | ({"augment"} if ctx.training else set()))
        self.env_steps = 0
        self.ended = 0
        self.episodes: list[tuple[float, int]] = []
        self.start()

    def start(self) -> None:
        """Reset the engine and act from its initial state."""
        n = self.ctx.num_envs
        self._epoch = self.ctx.dm._engine_epoch
        self.env_id = torch.arange(n)
        self._ret, self._len = torch.zeros(n), torch.zeros(n)
        self._observe(self.ctx.engine.reset())

    def _observe(self, values: tuple[Any, ...]) -> None:
        ctx = self.ctx
        self.cur = ctx.dm._stream_pass({self._keys[i]: values[i] for i in self._state}, self._act_phases)
        batch = ctx.dm._collate_env_axis(_to(self.cur, ctx.device))
        self.atoms = ctx.adapter.act(batch, deterministic=self.deterministic, generator=ctx.generator)

    def atom_for(self, item_key: str) -> torch.Tensor:
        """The latest captured atom that feeds back under ``item_key``."""
        atom = next(a for a, key in self.ctx.feedback.items() if key == item_key)
        return self.atoms[atom].detach().reshape(-1)

    def step(self) -> dict[str, Any]:
        ctx = self.ctx
        if self._epoch != ctx.dm._engine_epoch:
            self.start()
        step = ctx.engine.step(self.atoms[ACTION_ATOM])
        self.boundary = step.boundary.detach().to("cpu")
        feedback = {key: self.atoms[atom] for atom, key in ctx.feedback.items()}
        end = ctx.dm._stream_pass({**dict(zip(self._keys, step.end)), **feedback}, END_PHASES)
        row = build_row(self.cur, end)
        if self.episode_id is not None:
            row[self.episode_id] = self.env_id.clone()
        self._account(end)
        ctx.adapter.reset(step.boundary)
        self._observe(step.next)
        self.env_steps += ctx.num_envs
        return row

    def _account(self, end: dict[str, Any]) -> None:
        if self.return_key is not None:
            self._ret += detach_cpu(end[self.return_key]).reshape(-1).float()
        self._len += 1
        self.ended += int(self.boundary.sum())
        for i in torch.nonzero(self.boundary).reshape(-1).tolist():
            self.episodes.append((float(self._ret[i]), int(self._len[i])))
            self._ret[i], self._len[i] = 0.0, 0
        self.env_id = torch.where(self.boundary, self.env_id + self.ctx.num_envs, self.env_id)

    def drain_episodes(self) -> list[tuple[float, int]]:
        done, self.episodes = self.episodes, []
        return done

    def drain_metrics(self) -> dict[str, float]:
        done = self.drain_episodes()
        if not done:
            return {}
        returns, lengths = zip(*done)
        out = {"episode/length": sum(lengths) / len(lengths), "episode/count": float(len(done))}
        if self.return_key is not None:
            out["episode/return"] = sum(returns) / len(returns)
        return out
