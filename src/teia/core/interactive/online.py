"""Online episodic-metrics logging callback for the interactive regime.

Drains the runner's per-episode return/length records (``datamodule._runner.drain_episodes()``) on
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
    """Forward the runner's ended-episode records to ``pl_module.log``.

    ``window`` is the number of most-recent completed episodes averaged for each logged episode sum and for the
    length (a smoothing horizon, not a per-rollout reset). One series ``env/episode/<name>`` is logged per
    ``runner.episode_sums`` key (``<name>`` is the key without its ``item.`` prefix), the first one on the progress bar.
    """

    def __init__(self, window: int = 100) -> None:
        super().__init__()
        self.window = int(window)
        self._sums: dict[str, Deque[float]] = {}
        self._lengths: Deque[float] = deque(maxlen=self.window)
        self._total_episodes = 0

    def on_train_batch_end(self, trainer: Any, pl_module: Any, *args: Any, **kwargs: Any) -> None:
        runner = trainer.datamodule._runner
        for key in runner.episode_sums:
            self._sums.setdefault(key, deque(maxlen=self.window))
        records = runner.drain_episodes()
        for record in records:
            for key, total in record["sums"].items():
                self._sums[key].append(total)
            self._lengths.append(float(record["length"]))
        self._total_episodes += len(records)
        for i, (key, window) in enumerate(self._sums.items()):
            mean = sum(window) / len(window) if window else 0.0
            pl_module.log(f"env/episode/{key.split('.', 1)[-1]}", mean, on_step=True, on_epoch=False, prog_bar=i == 0)
        pl_module.log("env/episodic_length", sum(self._lengths) / len(self._lengths) if self._lengths else 0.0, on_step=True, on_epoch=False)
        pl_module.log("env/episodes", float(self._total_episodes), on_step=True, on_epoch=False)
