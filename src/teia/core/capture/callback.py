"""``CaptureCallback``: streaming val metrics during fit, whole-split capture on validate/test.

Both paths consume the ``atoms`` a validation/test step returns (``TeiaNetModule.step_atoms``,
teia:core/module/capture_map.md). The store lane follows the atom's contract spec type
(teia:core/eval.md#capture-store); nothing here interprets what an atom means.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from lightning.pytorch.callbacks import Callback

from teia.base.task import Blob, Ragged
from teia.core.capture.capture_io import capture_writer
from teia.core.capture.rasters import encode_blob
from teia.core.eval.streaming import LiveFrameSource, build_streaming_runner

__all__ = ["CaptureCallback"]


class CaptureCallback(Callback):
    def __init__(self, *, graph: Mapping[str, Any], ctx: Mapping[str, Any], max_rows: int | None, run_dir: Path) -> None:
        self.graph = dict(graph)
        self.ctx = dict(ctx)
        self.max_rows = max_rows
        self.run_dir = Path(run_dir)
        self._runner: Any = None
        self._writer: _AtomWriter | None = None

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        pl_module.capture_ctx = self.ctx

    # -- validation: streaming during fit, capture when standalone ------------------------------
    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        if _in_fit(trainer):
            self._runner = build_streaming_runner(self.graph, meta=trainer.datamodule.meta())
        else:
            self._writer = _AtomWriter(self.run_dir, "val", pl_module, trainer.datamodule.meta(), self.max_rows)

    def on_validation_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        atoms = outputs.get("atoms") if isinstance(outputs, Mapping) else None
        if atoms is None or trainer.sanity_checking:
            return
        if self._runner is not None:
            self._runner.update_batch(atoms)
        elif self._writer is not None:
            self._writer.write(atoms, outputs["n_samples"])

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self._runner is not None:
            for key, value in self._runner.compute().items():
                pl_module.log(f"val/{key}", float(value), prog_bar=(key == "fitness"), on_epoch=True, sync_dist=True)
            self._runner = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # -- test: capture ---------------------------------------------------------------------------
    def on_test_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._writer = _AtomWriter(self.run_dir, "test", pl_module, trainer.datamodule.meta(), self.max_rows)

    def on_test_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        atoms = outputs.get("atoms") if isinstance(outputs, Mapping) else None
        if atoms is not None and self._writer is not None:
            self._writer.write(atoms, outputs["n_samples"])

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _in_fit(trainer: Any) -> bool:
    fn = trainer.state.fn
    return str(getattr(fn, "value", fn)).lower() == "fit"


class _AtomWriter:
    """Writes atom dicts to one split's store; declares each column from its first batch."""

    def __init__(self, run_dir: Path, split: str, module: Any, meta: Mapping[str, Any], max_rows: int | None) -> None:
        capture = module.capture_map
        self.specs: dict[str, Any] = {f"capture.{atom}": spec for atom, spec in capture.specs.items()}
        self.specs.update({f"batch.{field}": spec for field, spec in capture.target_specs.items()})
        self.index_columns = {
            f"{column.split('.')[0]}.{spec.index}" for column, spec in self.specs.items() if isinstance(spec, Ragged)
        }
        self.writer = capture_writer(run_dir, split)
        self.writer.set_meta(
            task=None if module.contract is None else f"{type(module.contract).__module__}.{type(module.contract).__name__}",
            **{key: _jsonable(value) for key, value in meta.items()},
        )
        self.max_rows = max_rows
        self.samples = 0
        self.declared: set[str] = set()
        self.routes: dict[str, dict[str, str]] = {}

    def write(self, atoms: Mapping[str, Any], n: int) -> None:
        """``n`` is the step's sample count; ragged index columns are offset by the samples written so far."""
        if self.max_rows is not None and self.samples >= self.max_rows:
            return
        for column, value in atoms.items():
            spec = self.specs.get(column)
            if isinstance(spec, Blob):
                self._write_blobs(column, value, spec.format)
            elif isinstance(value, list):
                self._declare(column, "rows")
                self.writer.append_rows(column, [np.asarray(item).tolist() for item in value])
            else:
                array = np.asarray(value)
                array = array.astype(np.uint8) if array.dtype == bool else array
                if column in self.index_columns:
                    array = array + self.samples
                self._declare(column, "numeric", shape=array.shape[1:], dtype=spec.dtype if getattr(spec, "dtype", None) else array.dtype)
                self.writer.append_numeric(column, array)
        self.samples += n

    def _write_blobs(self, column: str, values: list[Any], fmt: str) -> None:
        self._declare(column, "blob", ext=fmt)
        for i, value in enumerate(values):
            encode_blob(value, self.writer.blob_path(column, self.samples + i), fmt)
            self.writer.record_blob(column)

    def _declare(self, column: str, lane: str, **kwargs: Any) -> None:
        if column in self.declared:
            return
        {"rows": self.writer.declare_rows, "numeric": self.writer.declare_numeric, "blob": self.writer.declare_blob}[lane](column, **kwargs)
        self.declared.add(column)
        if column.startswith("capture."):
            route, atom = column.removeprefix("capture.").split(".", 1)
            self.routes.setdefault(route, {})[atom] = column

    def close(self) -> None:
        for route, atoms in self.routes.items():
            self.writer.declare_route(route, **atoms)
        self.writer.close()


def _jsonable(value: Any) -> Any:
    return np.asarray(value).tolist() if isinstance(value, (tuple, np.ndarray)) or hasattr(value, "tolist") else value
