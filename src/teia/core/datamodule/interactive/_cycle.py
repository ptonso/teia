"""The stream-mode cycle (teia:core/datamodule.md#stream-mode), shared by the training runners and eval.

One ``step()`` advances every environment once: the engine steps with the action captured at the current state, the
``end`` item pass turns the signals into the transition's targets, the runner resets the finished environments from
the configured ``boundary`` items, the episodic hooks run, the ``next`` pass builds the state to act from, the
netmodule's capture slice acts, and the finished row is returned. Core names no field: the engine's signals, the
task contract's ``engine_atom`` and ``feedback`` map and the configured runner keys are the only vocabulary.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.core.datamodule.interactive._tensor_ops import detach_cpu
from teia.core.datamodule.interactive.runner_base import RunnerContext

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
    def __init__(self, ctx: RunnerContext, *, deterministic: bool) -> None:
        self.ctx, self.deterministic = ctx, deterministic
        self._keys = ctx.dm._engine_rec.out_key
        self._state = [i for i, (_, _, role) in enumerate(ctx.engine.spec().outputs) if role == "state"]
        self._act_phases = frozenset({"preprocess"} | ({"augment"} if ctx.training else set()))
        self.env_steps = 0
        self.ended = 0
        self.episodes: list[dict[str, Any]] = []
        self.start()

    def start(self) -> None:
        """Reset the engine and the episodic state, then act from the initial state."""
        n = self.ctx.num_envs
        self._epoch = self.ctx.dm._engine_epoch
        self.env_id = torch.arange(n)
        self._sums = {key: torch.zeros(n) for key in self.ctx.episode_sums}
        self._len = torch.zeros(n)
        self._idle: torch.Tensor | None = None
        self.ctx.dm._episodic_reset(torch.ones(n, dtype=torch.bool))
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

    def _ended(self, end: dict[str, Any]) -> torch.Tensor:
        """The boundary mask: the OR of the configured boundary items, minus idle environments."""
        flags = torch.stack([end[key].reshape(self.ctx.num_envs).bool() for key in self.ctx.boundary])
        ended = flags.any(0)
        return ended if self._idle is None else ended & ~self._idle

    def step(self) -> dict[str, Any]:
        ctx = self.ctx
        if self._epoch != ctx.dm._engine_epoch:
            self.start()
        signals = ctx.engine.step(self.atoms[ctx.engine_atom])
        feedback = {key: self.atoms[atom] for atom, key in ctx.feedback.items()}
        end = ctx.dm._stream_pass({**dict(zip(self._keys, signals)), **feedback}, END_PHASES)
        ended = self._ended(end)
        if ctx.valid is not None:
            end[ctx.valid] = torch.ones_like(ended) if self._idle is None else ~self._idle
        row = build_row(self.cur, end)
        if ctx.episode_id is not None:
            row[ctx.episode_id] = self.env_id.clone()
        self._account(end, ended.detach().to("cpu"))
        mask = self._reset_mask(ended)
        if bool(mask.any()):
            signals = ctx.engine.reset(mask)
        ctx.dm._episodic_reset(mask)
        ctx.dm._episodic_advance()
        ctx.adapter.reset(mask)
        self._observe(signals)
        self.env_steps += ctx.num_envs
        return row

    def _reset_mask(self, ended: torch.Tensor) -> torch.Tensor:
        """``per_env`` resets what ended; ``all_done`` idles ended environments until every one has ended."""
        if self.ctx.reset_policy == "per_env":
            return ended
        self._idle = ended if self._idle is None else self._idle | ended
        if bool(self._idle.all()):
            self._idle = None
            return torch.ones_like(ended)
        return torch.zeros_like(ended)

    def _account(self, end: dict[str, Any], boundary: torch.Tensor) -> None:
        for key, total in self._sums.items():
            total += detach_cpu(end[key]).reshape(-1).float()
        self._len += 1
        self.ended += int(boundary.sum())
        flags = torch.stack([detach_cpu(end[key]).reshape(-1).bool() for key in self.ctx.boundary])
        which = flags.int().argmax(0)
        for i in torch.nonzero(boundary).reshape(-1).tolist():
            self.episodes.append(
                {"sums": {key: float(total[i]) for key, total in self._sums.items()}, "length": int(self._len[i]), "ended_by": int(which[i])}
            )
            for total in self._sums.values():
                total[i] = 0.0
            self._len[i] = 0
        self.env_id = torch.where(boundary, self.env_id + self.ctx.num_envs, self.env_id)

    def drain_episodes(self) -> list[dict[str, Any]]:
        done, self.episodes = self.episodes, []
        return done

    def drain_metrics(self) -> dict[str, float]:
        done = self.drain_episodes()
        if not done:
            return {}
        count = float(len(done))
        out = {"episode/length": sum(e["length"] for e in done) / count, "episode/count": count}
        for key in self.ctx.episode_sums:
            out[f"episode/{key.split('.', 1)[-1]}"] = sum(e["sums"][key] for e in done) / count
        for i, key in enumerate(self.ctx.boundary):
            out[f"episode/ended_by/{key.split('.', 1)[-1]}"] = sum(e["ended_by"] == i for e in done) / count
        return out
