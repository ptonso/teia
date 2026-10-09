"""Per-step collection state, shared by the async Runner's inline and threaded modes.

One ``step_once()`` = one env step through the item subgraph (one cached policy forward) + one
``stream_reshape.add``. Reshape mutation and episode-log bookkeeping run under the supplied lock
so the threaded collector and the learner that drains ``stream_reshape.sample_iter()`` do not
race. Only off-policy (replay) reshape nodes are driven here — they store ``next_obs`` per step
and the learner trains on already-normalized observations read back from the buffer, so the
observation-transform running stats are only ever mutated on the collector side.
"""

from __future__ import annotations

import threading

import torch

from teia.core.datamodule.interactive._tensor_ops import detach_cpu
from teia.core.datamodule.interactive.runner_base import RunnerContext
from teia.core.datamodule.interactive.sync import _next_obs_fields, _normalize_only, _obs_seed


class CollectState:
    """Drives the env through the item subgraph, appending items to an off-policy reshape node."""

    def __init__(self, ctx: RunnerContext, lock: threading.Lock) -> None:
        self._ctx = ctx
        self._lock = lock
        self.raw_obs = ctx.stream_reader.reset()
        self.prev_done = torch.zeros(ctx.num_envs, dtype=torch.bool)
        self.env_step = 0
        self.episode_log = ctx.episode_log if ctx.training else None
        self.ep_return = torch.zeros(ctx.num_envs)
        self.ep_length = torch.zeros(ctx.num_envs)

    def step_once(self) -> None:
        ctx = self._ctx
        dm, reader = ctx.dm, ctx.stream_reader
        workspace = dm._item_workspace(_obs_seed(self.raw_obs, ctx.ctx), augment=False)
        env_action = detach_cpu(workspace["item.env_action"])
        step = reader.step(env_action)

        mask = (~self.prev_done).to(torch.float32)
        episode_start = self.prev_done.to(torch.float32)
        done = (step.terminated | step.truncated).to(torch.bool)

        item: dict = {k: v for k, v in workspace.items() if k.startswith("item.")}
        item["item.reward"] = step.reward
        item["item.terminated"] = step.terminated
        item["item.truncated"] = step.truncated
        item["item.mask"] = mask
        item["item.episode_start"] = episode_start
        next_ws = _normalize_only(dm, step.next_obs, training=False)
        item.update(_next_obs_fields(next_ws))

        with self._lock:
            ctx.stream_reshape.add(item)
            if self.episode_log is not None:
                self.ep_return += step.reward.detach().to(self.ep_return.dtype).reshape(-1) * mask
                self.ep_length += mask
                for i in torch.nonzero(done & (mask > 0)).reshape(-1).tolist():
                    self.episode_log.append((float(self.ep_return[i]), int(self.ep_length[i])))
                    self.ep_return[i] = 0.0
                    self.ep_length[i] = 0.0

        self.raw_obs = step.next_obs
        self.prev_done = done
        self.env_step += ctx.num_envs
