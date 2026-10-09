"""Online episodic-metrics logging callback for the interactive regime.

Drains the collection iterable's per-episode return/length log (``module._rl_episode_log``) on
every train batch into a rolling window and forwards the window mean to Lightning logging, so
episodic return shows up on the progress bar and any attached logger without the datamodule
needing direct logger access. Logs the rolling-window mean on every train batch (not just drain
batches) since episodes finish in bursts; otherwise the progress bar holds a stale value between
rollouts, reading as "training stalled" even while the return is moving.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

from lightning.pytorch.callbacks import Callback


class OnlineMetricsCallback(Callback):
    """Forward the collector's completed-episode metrics to ``pl_module.log``.

    ``window`` is the number of most-recent completed episodes averaged for the logged
    return/length (a smoothing horizon, not a per-rollout reset).
    """

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self.window = int(window)
        self._returns: Deque[float] = deque(maxlen=self.window)
        self._lengths: Deque[float] = deque(maxlen=self.window)
        self._total_episodes = 0

    def on_train_batch_end(self, trainer: Any, pl_module: Any, *args: Any, **kwargs: Any) -> None:
        log: list[tuple[float, int]] | None = getattr(pl_module, "_rl_episode_log", None)
        if log:
            for r, length in log:
                self._returns.append(float(r))
                self._lengths.append(float(length))
            self._total_episodes += len(log)
            log.clear()
        if not self._returns:
            pl_module.log("env/episodic_return", 0.0, on_step=True, on_epoch=False, prog_bar=True)
            pl_module.log("env/episodic_length", 0.0, on_step=True, on_epoch=False)
            pl_module.log("env/episodes", 0.0, on_step=True, on_epoch=False)
            return
        count = len(self._returns)
        pl_module.log("env/episodic_return", sum(self._returns) / count, on_step=True, on_epoch=False, prog_bar=True)
        pl_module.log("env/episodic_length", sum(self._lengths) / count, on_step=True, on_epoch=False)
        pl_module.log("env/episodes", float(self._total_episodes), on_step=True, on_epoch=False)


__all__ = ["OnlineMetricsCallback"]
