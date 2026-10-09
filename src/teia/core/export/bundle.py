from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from teia.core.export.preprocess_slice import PreprocessStep
from teia.core.utils import ensure_dir, write_json

BUNDLE_VERSION = 3


@dataclass(slots=True)
class ExportArtifact:
    source: Path
    relpath: str


@dataclass(slots=True)
class BundlePaths:
    root: Path
    metadata_dir: Path
    runtime_dir: Path
    onnx_dir: Path

    @classmethod
    def create(cls, root: Path) -> BundlePaths:
        resolved_root = ensure_dir(root)
        return cls(
            root=resolved_root,
            metadata_dir=ensure_dir(resolved_root / "metadata"),
            runtime_dir=ensure_dir(resolved_root / "runtime"),
            onnx_dir=ensure_dir(resolved_root / "onnx"),
        )


@dataclass(slots=True)
class InputSpec:
    """One named ONNX input, tracing back to one ``batch.<field>`` model input."""

    name: str
    payload_key: str
    dtype: str = "float32"
    shape: list[Any] = field(default_factory=lambda: ["batch", "..."])
    payload: dict[str, Any] = field(default_factory=lambda: {"kind": "value"})
    """What the export client must send for this field (``{"kind": "value"}`` or
    ``{"kind": "window", **reshape.payload_spec()}``), from ``resolve_chain_for``'s 3-tuple."""


@dataclass(slots=True)
class BundleContract:
    task: str
    exporter_family: str
    postprocess: dict[str, Any]
    inputs: list[InputSpec] = field(default_factory=list)
    preprocess_chains: dict[str, list[PreprocessStep]] = field(default_factory=dict)
    """Per-field ``batch.<field>`` preprocess chain (from ``preprocess_slice``); ``assemble.py``
    collects each step's kernel into the bundle rather than serializing this field."""
    labels: list[str] = field(default_factory=list)
    routes: list[str] | None = None
    """Activation-route names to include (``model._activation_records`` names); ``None`` selects
    all. Read from ``export.routes`` config — e.g. an interactive recipe restricting to ``["policy"]``."""
    runtime_family: str = "onnxruntime"
    entrypoint: str = "runtime/run.py"
    model_relpath: str = "onnx/model.onnx"
    outputs: list[dict[str, Any]] = field(default_factory=list)
    dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)
    runtime_options: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ExportArtifact] = field(default_factory=list)
    extra_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    """JSON-able bundle extras collected from every data node's ``export_artifacts()``, keyed by
    filename stem (e.g. ``{"schema": {...}}`` writes ``metadata/schema.json``)."""


def build_manifest(contract: BundleContract) -> dict[str, Any]:
    artifact_map: dict[str, str] = {"model": contract.model_relpath}
    for artifact in contract.artifacts:
        artifact_map[Path(artifact.relpath).stem] = artifact.relpath
    specs = {
        "task": "metadata/task.json",
        "labels": "metadata/labels.json",
        "preprocess": "metadata/preprocess.json",
        "postprocess": "metadata/postprocess.json",
    }
    for key in contract.extra_metadata:
        specs[key] = f"metadata/{key}.json"
    return {
        "bundle_version": BUNDLE_VERSION,
        "task": contract.task,
        "runtime": {
            "family": contract.runtime_family,
            "entrypoint": contract.entrypoint,
        },
        "artifacts": artifact_map,
        "specs": specs,
    }


def build_task_spec(contract: BundleContract) -> dict[str, Any]:
    return {
        "task": contract.task,
        "exporter_family": contract.exporter_family,
        "runtime_family": contract.runtime_family,
        "inputs": [
            {
                "name": item.name,
                "payload_key": item.payload_key,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "payload": dict(item.payload),
            }
            for item in contract.inputs
        ],
        "outputs": [dict(item) for item in contract.outputs],
        "dynamic_axes": {
            name: {str(axis): value for axis, value in axes.items()}
            for name, axes in contract.dynamic_axes.items()
        },
        "runtime_options": dict(contract.runtime_options),
    }


def build_preprocess_spec(contract: BundleContract) -> dict[str, Any]:
    """Lossless, auto-derived per-field preprocess summary (debug/inspection only — the emitted
    ``runtime/preprocess.py`` is the executable source of truth)."""
    return {
        field_name: [dict(step.params) for step in chain] for field_name, chain in contract.preprocess_chains.items()
    }


def write_bundle_spec(paths: BundlePaths, contract: BundleContract) -> None:
    write_json(paths.root / "manifest.json", build_manifest(contract))
    write_json(paths.metadata_dir / "task.json", build_task_spec(contract))
    write_json(paths.metadata_dir / "labels.json", {"names": list(contract.labels)})
    write_json(paths.metadata_dir / "preprocess.json", build_preprocess_spec(contract))
    write_json(paths.metadata_dir / "postprocess.json", dict(contract.postprocess))
    for key, payload in contract.extra_metadata.items():
        write_json(paths.metadata_dir / f"{key}.json", payload)
    for artifact in contract.artifacts:
        dest = paths.root / artifact.relpath
        ensure_dir(dest.parent)
        shutil.copy2(artifact.source, dest)


def write_bundle_readme(paths: BundlePaths, contract: BundleContract) -> None:
    content = (
        "# Teia Export Bundle\n\n"
        f"- task: `{contract.task}`\n"
        f"- runtime: `{contract.runtime_family}`\n"
        f"- exporter family: `{contract.exporter_family}`\n\n"
        "The bundle contract is described by `manifest.json` and the files in `metadata/`.\n"
    )
    (paths.root / "README.md").write_text(content, encoding="utf-8")
