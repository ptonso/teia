"""
Episode aggregate metric: the ended-episode records of a stream-mode eval cycle reduced to return, length and termination.

Source: common knowledge

Description:
  Reads one entry per ended episode (``episodes.<name>`` keys): the episode sum named ``return_name`` (default
  ``reward``), ``length`` and ``ended_by``, the index of the ``runner.boundary`` key that ended the episode.
  ``terminated_rate`` is the fraction of episodes ended by boundary key ``terminated_index`` (default 0, the
  termination key rather than a time limit). A pass with no ended episode (a dry run's single step) reports zero
  episodes and NaN statistics. A terminal state is success in a goal-reaching env and failure in a balancing one,
  and the records carry no success signal, so no success rate is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from teia.base.eval import Metric

__all__ = ["EpisodeAggregate"]


class EpisodeAggregate(Metric):
    """Mean ``episode_return``, ``episode_length`` and ``terminated_rate`` over ended episodes."""

    def __init__(self, return_name: str = "reward", terminated_index: int = 0) -> None:
        self.return_name, self.terminated_index = return_name, int(terminated_index)

    def from_source(self, source: Any, *, length: Any, ended_by: Any, **sums: Any) -> dict[str, Any]:
        del source
        returns = np.asarray(sums[self.return_name], dtype=float).reshape(-1)
        length = np.asarray(length, dtype=float).reshape(-1)
        terminated = np.asarray(ended_by).reshape(-1) == self.terminated_index
        if not len(returns):
            nan = float("nan")
            return {"episodes": 0, "episode_return": nan, "episode_return_std": nan, "episode_length": nan, "terminated_rate": nan, "rows": []}
        rows = [
            {"episode_id": i, "episode_return": float(r), "episode_length": int(n), "terminated": bool(t)}
            for i, (r, n, t) in enumerate(zip(returns, length, terminated))
        ]
        return {
            "episodes": len(rows),
            "episode_return": float(returns.mean()),
            "episode_return_std": float(returns.std()),
            "episode_length": float(length.mean()),
            "terminated_rate": float(terminated.mean()),
            "rows": rows,
        }
