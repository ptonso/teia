"""Filesystem-root + path-provenance helpers shared by every filesystem-scanning data node.

Root resolution (``resolve_root``) is format-agnostic and reused by frame and image readers alike; the
split/class extraction turns a root-relative path into its ``(split, class_parts)`` for directory-tree
datasets. No modality: a recipe's reader owns the scan, these are the shared primitives.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".flac", ".mp3", ".ogg", ".wav"}
LABEL_EXTENSIONS = {".txt", ".json", ".xml"}
ClassNameMode = Literal["basename", "full_path"]

_SPLIT_DIR_NAMES = {"train", "val", "valid", "validation", "test"}
_SPLIT_CANONICAL = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
CLASS_NAME_MODES = {"basename", "full_path"}


def canonical_split_id(name: str) -> str:
    """Canonical split id (``train``/``val``/``test``) for a directory name."""
    return _SPLIT_CANONICAL.get(str(name).strip().lower(), "train")


def resolve_root(root: str | Path | None, *, data_root: str | Path | None, project_dir: str | Path | None) -> Path:
    """Resolve a recipe ``root`` against ``data_root``/``project_dir`` (mirrors the former resolver)."""
    if root is not None and Path(root).is_absolute():
        return Path(root)
    base = Path(data_root) if data_root is not None else Path(project_dir or ".")
    if not base.is_absolute():
        base = (Path(project_dir or ".") / base).resolve()
    if root is None:
        return base
    parts = Path(root).parts
    if parts and parts[0].lower() == "datamodule":
        parts = parts[1:]
    return (base / Path(*parts)).resolve() if parts else base


def iter_files(root: Path, extensions: set[str]) -> list[Path]:
    """Sorted files under ``root`` (recursive) whose suffix is in ``extensions``."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def iter_images(root: Path) -> list[Path]:
    """Sorted image files under ``root`` (recursive)."""
    return iter_files(root, IMAGE_EXTENSIONS)


def split_and_class_parts(rel: Path) -> tuple[str, list[str]]:
    """``(split, class_parts)`` from a root-relative image path.

    A leading canonical split dir sets the split (default ``train``); a trailing ``images`` dir is
    dropped; the remaining dirs are the class parts. Matches the class-dir layout.
    """
    dirs = list(rel.parts[:-1])
    split = "train"
    if dirs and dirs[0].lower() in _SPLIT_DIR_NAMES:
        split = canonical_split_id(dirs[0])
        dirs = dirs[1:]
    if dirs and dirs[-1].lower() == "images":
        dirs = dirs[:-1]
    return split, dirs


def resolve_class_name(class_parts: list[str], *, class_name_mode: ClassNameMode) -> str | None:
    """Class name from ``class_parts`` per ``class_name_mode`` (basename/full_path)."""
    if not class_parts:
        return None
    if class_name_mode == "basename":
        return class_parts[-1]
    if class_name_mode == "full_path":
        return Path(*class_parts).as_posix()
    raise ValueError(f"class_name_mode must be one of {sorted(CLASS_NAME_MODES)}; got {class_name_mode!r}.")


def has_top_level_splits(root: Path) -> bool:
    """True if ``root`` contains train/val/test split subdirectories."""
    return any((root / name).is_dir() for name in _SPLIT_DIR_NAMES)
