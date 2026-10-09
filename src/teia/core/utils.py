from __future__ import annotations

import importlib
import json
import re
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def copy_tree(source_root: str | Path, target_root: str | Path) -> list[Path]:
    source = Path(source_root)
    target = Path(target_root)
    written: list[Path] = []
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        target_path = target / source_path.relative_to(source)
        ensure_dir(target_path.parent)
        shutil.copyfile(source_path, target_path)
        written.append(target_path)
    return written


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def import_string(path: str):
    module_name, _, attr_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid import path: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = deepcopy(base)
        for key, value in overlay.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(overlay)


def strip_internal_metadata(payload: Any) -> Any:
    from collections.abc import Mapping, Sequence
    if isinstance(payload, Mapping):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(key, str) and key.startswith("__"):
                continue
            cleaned[key] = strip_internal_metadata(value)
        return cleaned
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [strip_internal_metadata(item) for item in payload]
    return deepcopy(payload)


def disable_pretrained_loading(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "pretrained" and isinstance(value, bool):
                cleaned[key] = False
            else:
                cleaned[key] = disable_pretrained_loading(value)
        return cleaned
    if isinstance(payload, list):
        return [disable_pretrained_loading(item) for item in payload]
    return deepcopy(payload)


def get_dot_path(mapping: dict[str, Any], dotted_path: str) -> Any:
    """Value at a Hydra-style dotted path; a numeric part indexes a list (``schedulers.0.total_iters``)."""
    current: Any = mapping
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(dotted_path)
    return current


def parse_override(override: str) -> tuple[str, Any]:
    key, sep, raw_value = override.partition("=")
    if not sep:
        raise ValueError(f"Overrides must look like key=value, got: {override}")
    return key, yaml.safe_load(raw_value)


def read_yaml(path: str | Path) -> dict[str, Any]:
    content = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    return data or {}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _render_log_value(key: str, value: Any, fields: dict[str, Any]) -> str:
    candidate = _coerce_pathish(value)
    if candidate is not None:
        run_dir = _coerce_pathish(fields.get("run_dir"))
        if key == "checkpoint" and run_dir is not None:
            relative_to_run_dir = _relativize_to_base(candidate, run_dir)
            if relative_to_run_dir is not None:
                return relative_to_run_dir
        return _relativize_log_path(candidate)
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_pathish(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return None
    return None


def _relativize_log_path(path: Path) -> str:
    relative = _relativize_to_base(path, Path.cwd())
    if relative is not None:
        return relative
    return str(path)


def _relativize_to_base(path: Path, base: Path) -> str | None:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        resolved_base = base.resolve()
    except Exception:
        resolved_base = base
    try:
        relative = resolved.relative_to(resolved_base)
    except ValueError:
        return None
    rendered = str(relative)
    return rendered or "."


def teia_log(event: str, /, **fields: Any) -> None:
    message = f"[teia] {event}"
    rendered_fields: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        rendered_value = _render_log_value(key, value, fields)
        rendered_fields.append(f"{key}={rendered_value}")
    if rendered_fields:
        message = f"{message}: {' '.join(rendered_fields)}"
    print(message, flush=True)
