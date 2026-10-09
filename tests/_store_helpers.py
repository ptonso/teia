"""Test helpers: materialize a capture store from a literal, and compose a task's evalmodule graph.

``write_store`` mirrors the capture writer's lanes (teia:core/eval.md#capture-store): fixed-shape
numeric rows go to the numeric lane, lists of strings or variable-size items go to the rows lane,
and every ``capture.<route>.<atom>`` column is indexed under its route.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import importlib

import yaml

from teia.core.capture.store import CaptureWriter, store_dir
from teia.core.eval.runner import split_evalmodule
from teia.core.runtime.composer import compose_overlay_config

_SHARED_RUN = Path(importlib.import_module("teia.core").__file__).resolve().parents[1] / "conf/evalmodule/_shared/run.yaml"


def write_store(run_dir: str | Path, split: str, columns: dict[str, Any], **meta: Any) -> Path:
    writer = CaptureWriter(store_dir(run_dir, split), split=split)
    routes: dict[str, dict[str, str]] = {}
    for column, values in columns.items():
        rows = list(values)
        if rows and (isinstance(rows[0], str) or _ragged(rows)):
            writer.declare_rows(column)
            writer.append_rows(column, [np.asarray(row).tolist() if not isinstance(row, str) else row for row in rows])
        else:
            array = np.asarray(rows)
            array = array.reshape(len(rows), -1) if array.ndim == 1 else array
            writer.declare_numeric(column, shape=array.shape[1:], dtype=str(array.dtype if array.dtype != np.float64 else "float32"))
            writer.append_numeric(column, array)
        if column.startswith("capture."):
            route, atom = column.removeprefix("capture.").split(".", 1)
            routes.setdefault(route, {})[atom] = column
    for route, atoms in routes.items():
        writer.declare_route(route, **atoms)
    writer.set_meta(**meta)
    return writer.close()


def load_evalmodule(task: str, *, run_nodes: bool = False) -> dict[str, Any]:
    """The composed ``evalmodule`` nodes of ``evalmodule=<task>/default`` (without the shared run nodes by default,
    which read logger history a store-only fixture has none of)."""
    with tempfile.TemporaryDirectory() as tmp:
        composed, _ = compose_overlay_config(project_dir=Path(tmp), overrides=[f"evalmodule={task}/default", "data_root=."])
    _, graph = split_evalmodule(composed["evalmodule"])
    if not run_nodes:
        for name in yaml.safe_load(_SHARED_RUN.read_text()):
            graph.pop(name, None)
    return graph


def _ragged(rows: list[Any]) -> bool:
    shapes = {np.asarray(row).shape for row in rows}
    return len(shapes) > 1
