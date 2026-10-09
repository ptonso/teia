from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from teia.core.utils import ensure_dir, teia_log, read_yaml, write_yaml

from lightning.pytorch.callbacks import Timer
from lightning.pytorch.trainer.states import RunningStage


WALLTIME_REPORT_FILENAME = "walltime.yaml"
SCHEMA_VERSION = 1

# stage name -> Lightning RunningStage used to query Timer wall time
_STAGES = {
    "train": RunningStage.TRAINING,
    "validate": RunningStage.VALIDATING,
    "test": RunningStage.TESTING,
    "predict": RunningStage.PREDICTING,
}


class WalltimeCallback(Timer):
    """Emit ``eval/walltime.yaml`` with Lightning-native wall time + throughput.

    Subclasses Lightning's ``Timer`` (the canonical per-stage stopwatch) and layers batch /
    sample counters on top to derive iterations-per-second and samples-per-second for the
    train, validate, test (inference), and predict stages. The artifact is merged across the
    separate trainers that fit and post-train test/infer use, so one file accumulates both the
    training throughput and the inference benchmark for a run.
    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self._batches: dict[str, int] = {key: 0 for key in _STAGES}
        self._samples: dict[str, int] = {key: 0 for key in _STAGES}
        self._epochs = 0

    def on_train_epoch_end(self, trainer, *args: Any, **kwargs: Any) -> None:
        super().on_train_epoch_end(trainer, *args, **kwargs)
        self._epochs += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self._count("train", batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if not getattr(trainer, "sanity_checking", False):
            self._count("validate", batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._count("test", batch)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        super().on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self._count("predict", batch)

    def on_fit_end(self, trainer, pl_module) -> None:
        super().on_fit_end(trainer, pl_module)
        self._write(trainer)

    def on_test_end(self, trainer, pl_module) -> None:
        super().on_test_end(trainer, pl_module)
        self._write(trainer)

    def on_predict_end(self, trainer, pl_module) -> None:
        super().on_predict_end(trainer, pl_module)
        self._write(trainer)

    def _count(self, stage: str, batch: Any) -> None:
        self._batches[stage] += 1
        self._samples[stage] += _infer_batch_size(batch)

    def _write(self, trainer) -> None:
        if not self.enabled or getattr(trainer, "global_rank", 0) != 0:
            return
        stages = {
            stage: self._stage_entry(stage, running_stage)
            for stage, running_stage in _STAGES.items()
            if self._batches[stage] > 0
        }
        if not stages:
            return
        path = Path(str(getattr(trainer, "default_root_dir", Path.cwd()))) / "eval" / WALLTIME_REPORT_FILENAME
        ensure_dir(path.parent)
        payload = read_yaml(path) if path.exists() else {}
        payload["schema_version"] = SCHEMA_VERSION
        payload["generated_by"] = f"{type(self).__module__}.{type(self).__name__}"
        payload.setdefault("stages", {})
        payload["stages"].update(stages)
        write_yaml(path, payload)
        teia_log("walltime report", artifact=path, stages=sorted(stages))

    def _stage_entry(self, stage: str, running_stage) -> dict[str, Any]:
        seconds = float(self.time_elapsed(running_stage))
        batches = self._batches[stage]
        samples = self._samples[stage]
        entry: dict[str, Any] = {
            "wall_seconds": round(seconds, 3),
            "wall_human": str(timedelta(seconds=int(seconds))),
            "batches": batches,
            "iters_per_sec": round(batches / seconds, 3) if seconds > 0 else None,
        }
        if samples > 0:
            entry["samples"] = samples
            entry["samples_per_sec"] = round(samples / seconds, 3) if seconds > 0 else None
        if stage == "train" and self._epochs > 0:
            entry["epochs"] = self._epochs
            entry["sec_per_epoch"] = round(seconds / self._epochs, 3)
        return entry


def _infer_batch_size(batch: Any) -> int:
    """Best-effort sample count: leading dim of the first tensor-like field in the batch."""
    if hasattr(batch, "shape"):
        return int(batch.shape[0]) if batch.shape else 0
    if isinstance(batch, dict):
        values: Any = batch.values()
    elif isinstance(batch, (tuple, list)):
        values = batch
    else:
        return 0
    for value in values:
        size = _infer_batch_size(value)
        if size:
            return size
    return 0
