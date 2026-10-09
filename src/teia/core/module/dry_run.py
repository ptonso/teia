from __future__ import annotations

import logging
from typing import Any, get_origin

import torch

log = logging.getLogger(__name__)


def dry_run_pipeline(module, dm) -> None:
    """Smoke-test the generated forward pass on a synthetic batch; logs pred shapes."""
    batch_cls = dm.batch_type()
    meta: dict = dm.batch_meta()
    B = int(getattr(dm, "batch_size", None) or 2)

    fields: dict = {}
    for field_name, shape in meta.items():
        if isinstance(shape, (tuple, list)) and all(isinstance(s, int) for s in shape):
            fields[field_name] = torch.zeros(B, *shape)
        else:
            fields[field_name] = torch.zeros(B)

    for field_name in batch_cls._fields:
        if field_name not in fields:
            fields[field_name] = _synthetic_missing_field(batch_cls, field_name, B)

    batch = batch_cls(**fields)
    module.example_input_array = (batch,)
    module.eval()
    try:
        with torch.no_grad():
            try:
                pred = module.forward(batch)
            except Exception as exc:
                raise RuntimeError(f"dry_run_pipeline failed: {exc}") from exc

        for field_name in pred._fields:
            val = getattr(pred, field_name)
            shape_str = str(val.shape) if hasattr(val, "shape") else type(val).__name__
            log.info("dry_run | pred.%s: %s", field_name, shape_str)
    finally:
        module.train()


def _synthetic_missing_field(batch_cls: type, field_name: str, batch_size: int) -> Any:
    defaults = getattr(batch_cls, "_field_defaults", {}) or {}
    if field_name in defaults and defaults[field_name] is not None:
        return defaults[field_name]

    annotation = (getattr(batch_cls, "__annotations__", {}) or {}).get(field_name)
    origin = get_origin(annotation)
    annotation_text = str(annotation).lower()
    if origin is list or "list" in annotation_text:
        return []
    if origin is dict or "dict" in annotation_text:
        return {}
    if origin is tuple or "tuple" in annotation_text:
        return ()
    return torch.zeros(batch_size)
