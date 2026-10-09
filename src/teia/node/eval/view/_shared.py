"""
Shared rendering helpers for ``teia.node.eval.view.*`` components.

Source: common knowledge

Description:
  Private infrastructure, not a component: no ``_target_``, no ``conf/eval/view/*.yaml`` leaf.
  Every concrete view module does ``from teia.node.eval.view._shared import ...``.

These helpers operate directly on primitive labels, arrays and config values. The styling
config lives on each View's constructor, and the data (labels, arrays) arrives through
``render(**inputs)``.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (Agg backend must be set first)
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Okabe-Ito colorblind-safe palette, applied process-wide.
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_PALETTE)

__all__ = [
    "plt",
    "np",
    "yaml",
    "_PALETTE",
    "_compact_label",
    "_wrap_path_label",
    "_wrap_label",
    "_display_label",
    "_display_labels",
    "_line_count",
    "_all_same_nonzero",
    "_scatter_can_label_directly",
    "_scatter_label_order",
    "_scatter_label_y",
    "_apply_report_style",
    "_balanced_chunks",
    "_limits",
    "_page_filename",
    "_page_title",
    "slugify",
    "write_description_sidecar",
]

def _compact_label(value: Any) -> str:
    text = str(value)
    if "/" not in text:
        return text
    parts = text.split("/")
    if len(parts) <= 2:
        return text
    return f"{parts[0]}/.../{parts[-1]}"


def _wrap_path_label(text: str, *, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for index, part in enumerate(text.split("/")):
        segment = f"{part}/" if index < text.count("/") else part
        if not current:
            current = segment
        elif len(current) + len(segment) <= width:
            current += segment
        else:
            lines.append(current)
            current = segment
        while len(current) > width:
            trailing_slash = current.endswith("/")
            body = current[:-1] if trailing_slash else current
            wrapped = textwrap.wrap(body, width=width - int(trailing_slash), break_long_words=True, break_on_hyphens=False)
            if not wrapped:
                break
            lines.extend(wrapped[:-1])
            current = f"{wrapped[-1]}/" if trailing_slash else wrapped[-1]
    if current:
        lines.append(current)
    return lines


def _wrap_label(value: Any, *, width: int = 24, max_lines: int = 4) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    lines = _wrap_path_label(text, width=width) if "/" in text else textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = f"{lines[-1][: max(width - 3, 1)]}..."
    return "\n".join(lines)


def _display_label(value: Any, *, width: int = 24, max_lines: int = 4) -> str:
    return _wrap_label(_compact_label(value), width=width, max_lines=max_lines)


def _display_labels(values: list[Any], *, width: int = 24, max_lines: int = 4) -> list[str]:
    return [_display_label(value, width=width, max_lines=max_lines) for value in values]


def _line_count(label: str) -> int:
    return label.count("\n") + 1


def _all_same_nonzero(values: list[float]) -> bool:
    return bool(values) and all(float(value) > 0.0 for value in values) and len({float(value) for value in values}) == 1


def _scatter_can_label_directly(xs: list[float], ys: list[float], labels: list[str]) -> bool:
    rounded = {(round(float(x), 3), round(float(y), 3)) for x, y in zip(xs, ys, strict=False)}
    return len(labels) <= 8 and len(rounded) == len(labels)


def _scatter_label_order(xs: list[float], ys: list[float], labels: list[str]) -> list[int]:
    return sorted(range(len(labels)), key=lambda index: (-float(ys[index]), float(xs[index]), str(labels[index])))


def _scatter_label_y(rank: int, count: int) -> float:
    return 0.98 if count == 1 else 0.98 - rank * (0.96 / (count - 1))


def _apply_report_style(
    fig: Any,
    ax: Any,
    *,
    scatter: bool = False,
    bounded: bool = False,
    bounded_x: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    axis_pad = 0.025 if scatter else 0.0
    if ylim:
        ax.set_ylim(*ylim)
    elif bounded:
        ax.set_ylim(0.0 - axis_pad, 1.0 + axis_pad)
    if xlim:
        ax.set_xlim(*xlim)
    elif bounded_x:
        ax.set_xlim(0.0 - axis_pad, 1.0 + axis_pad)


def _balanced_chunks(values: list[Any], max_size: int) -> list[list[Any]]:
    if len(values) <= max_size:
        return [values]
    page_count = (len(values) + max_size - 1) // max_size
    base_size = len(values) // page_count
    extras = len(values) % page_count
    chunks: list[list[Any]] = []
    start = 0
    for page_index in range(page_count):
        size = base_size + int(page_index < extras)
        chunks.append(values[start : start + size])
        start += size
    return chunks


def _limits(values: list[float]) -> tuple[float, float]:
    low = min(float(value) for value in values)
    high = max(float(value) for value in values)
    if low == high:
        return (low - 0.5, high + 0.5)
    pad = (high - low) * 0.05
    return (low - pad, high + pad)


def _page_filename(filename: str, page_index: int, page_count: int) -> str:
    """Page 1 keeps the base filename; later pages get a ``__page_NN`` suffix inserted before
    the extension."""
    if page_count == 1:
        return filename
    path = Path(filename)
    return f"{path.stem}__page_{page_index + 1:02d}{path.suffix}"


def _page_title(title: str, page_index: int, page_count: int) -> str:
    if page_count == 1:
        return title
    return f"{title} (page {page_index + 1}/{page_count})"


def slugify(text: str) -> str:
    """Lowercase, non-alphanumeric runs collapsed to a single ``_``, edges stripped."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug or "plot"


def write_description_sidecar(
    out_dir: Path,
    stem: str,
    *,
    title: str,
    description: str | None,
    interpretation: str | None,
    blindspot: str | None,
) -> Path | None:
    """Writes ``{stem}.description.yaml`` when any of description/interpretation/blindspot is
    set. ``title`` is always included when a sidecar is written."""
    if description is None and interpretation is None and blindspot is None:
        return None
    payload: dict[str, str] = {"title": title}
    if description is not None:
        payload["description"] = description
    if interpretation is not None:
        payload["interpretation"] = interpretation
    if blindspot is not None:
        payload["blindspot"] = blindspot
    path = Path(out_dir) / f"{stem}.description.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
