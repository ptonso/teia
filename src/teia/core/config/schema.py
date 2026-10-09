from __future__ import annotations

from dataclasses import dataclass


#: Required composed groups. ``evalmodule`` and ``task`` are optional (a task-less run may skip eval).
TOP_LEVEL_GROUPS = (
    "datamodule",
    "netmodule",
    "trainer",
    "runtime",
    "callbacks",
    "loggers",
    "train",
    "test",
    "infer",
    "export",
)


@dataclass(slots=True)
class RuntimeConfig:
    seed: int | None = 42
    float32_matmul_precision: str | None = "high"
    determinism: dict[str, str] | None = None


@dataclass(slots=True)
class ExportConfig:
    checkpoint: str | None = None
    output_dir: str | None = None
    format: str = "onnx"
    opset: int = 18
    bundle_name: str = "model"
