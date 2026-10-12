"""
Episode aggregate metric: per-step rows grouped by episode into return, length and termination.

Source: common knowledge

Description:
  Groups captured per-step rows by ``episode_id`` and keeps the episodes that ended, those with a
  ``terminated`` or ``truncated`` row. Episodes still running when the capture stopped are
  dropped. ``terminated_rate`` is the fraction of ended episodes that reached a terminal state
  rather than a time limit. A terminal state is success in a goal-reaching env and failure in a
  balancing one, and the capture carries no success signal, so no success rate is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric

__all__ = ["EpisodeAggregate"]


class EpisodeAggregate(Metric):
    """Mean ``episode_return``, ``episode_length`` and ``terminated_rate`` over ended episodes. No constructor kwargs."""

    def from_source(self, source: Any, *, reward: Any, terminated: Any, truncated: Any, episode_id: Any, **_: Any) -> dict[str, Any]:
        del source
        reward = np.asarray(reward, dtype=float).reshape(-1)
        terminated = np.asarray(terminated).reshape(-1).astype(bool)
        ended = terminated | np.asarray(truncated).reshape(-1).astype(bool)
        episode_id = np.asarray(episode_id).reshape(-1)
        masks = {int(episode): episode_id == episode for episode in np.unique(episode_id[ended])}
        if not masks:
            raise ValueError("no captured episode ended; the capture holds partial episodes only.")
        rows = [
            {
                "episode_id": episode,
                "episode_return": float(reward[mask].sum()),
                "episode_length": int(mask.sum()),
                "terminated": bool(terminated[mask].any()),
            }
            for episode, mask in masks.items()
        ]
        return {
            "episodes": len(rows),
            "episode_return": float(np.mean([row["episode_return"] for row in rows])),
            "episode_return_std": float(np.std([row["episode_return"] for row in rows])),
            "episode_length": float(np.mean([row["episode_length"] for row in rows])),
            "terminated_rate": float(np.mean([row["terminated"] for row in rows])),
            "rows": rows,
        }
