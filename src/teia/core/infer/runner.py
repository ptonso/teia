from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from teia.base.task import Blob, Ragged
from teia.core.infer.writers import write_predictions_jsonl


def run_infer(
    *,
    config: dict[str, Any],
    context: dict[str, Any],
    datamodule: Any,
    module: Any,
    checkpoint: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Task-agnostic dataset inference (teia:core/infer.md).

    One ``trainer.predict`` loop yields ``capture.*`` atoms per batch through the netmodule's
    ``capture:`` map, decoded with ``infer.ctx`` plus the batch fields named by ``infer.restore``
    (e.g. ``ori_shape``, ``ratio_pad``) so predictions land in original coordinates. The atoms go to
    ``predictions.jsonl`` (one line per sample) and to the datamodule's writer nodes.
    """
    from teia.core.runtime.engine import run_predict_stage

    if not callable(getattr(datamodule, "write_predictions", None)):
        raise RuntimeError(
            f"{type(datamodule).__name__} does not support `teia infer` (no predict dataset); "
            "interactive runs are online rollouts."
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_cfg = config.get("infer") or {}
    static_ctx = dict(infer_cfg.get("ctx") or {})
    restore = list(infer_cfg.get("restore") or [])
    module._predict_ctx_builder = lambda batch: {**static_ctx, **{key: getattr(batch, key) for key in restore}}
    module.infer_batch_keys = datamodule.writer_batch_keys()
    try:
        stage = run_predict_stage(
            config=config,
            datamodule_obj=datamodule,
            module_obj=module,
            checkpoint=checkpoint,
            run_dir=Path(context["run_dir"]),
        )
    finally:
        module._predict_ctx_builder = None

    specs = {f"capture.{atom}": spec for atom, spec in module.capture_map.specs.items()}
    index_of = {col: f"capture.{spec.index}" for col, spec in specs.items() if isinstance(spec, Ragged)}
    atoms, n = concat_atoms(stage["result"] or [], set(index_of.values()))
    records = sample_records(atoms, n, index_of, blobs={col for col, spec in specs.items() if isinstance(spec, Blob)})
    io_outputs = datamodule.write_predictions(atoms=atoms, dst=output_dir / "dataset")
    predictions_path = write_predictions_jsonl(output_dir / "predictions.jsonl", records)
    return {"predictions_path": str(predictions_path), "records": records, "io_outputs": io_outputs}


def concat_atoms(steps: list[dict[str, Any]], index_columns: set[str]) -> tuple[dict[str, Any], int]:
    """Concatenate per-batch ``{"atoms", "n_samples"}`` outputs; ragged index columns become global."""
    out: dict[str, list[Any]] = {}
    offset = 0
    for step in steps:
        for column, value in step["atoms"].items():
            if isinstance(value, list):
                out.setdefault(column, []).extend(value)
                continue
            tensor = torch.as_tensor(value)
            out.setdefault(column, []).append(tensor + offset if column in index_columns else tensor)
        offset += int(step["n_samples"])
    return {col: torch.cat(parts) if parts and isinstance(parts[0], torch.Tensor) else parts for col, parts in out.items()}, offset


def sample_records(atoms: dict[str, Any], n: int, index_of: dict[str, str], *, blobs: set[str]) -> list[dict[str, Any]]:
    """One JSON-able record per sample; ragged atoms are sliced by their index, blobs are omitted."""
    records: list[dict[str, Any]] = [{"sample_id": i} for i in range(n)]
    for column, value in atoms.items():
        if column in blobs or column in index_of.values():
            continue
        if column in index_of:
            index = np.asarray(atoms[index_of[column]])
            items = value if isinstance(value, list) else np.asarray(value)
            for i, record in enumerate(records):
                rows = np.flatnonzero(index == i)
                record[column] = [np.asarray(items[r]).tolist() for r in rows]
        else:
            for i, record in enumerate(records):
                record[column] = np.asarray(value[i]).tolist()
    return records
