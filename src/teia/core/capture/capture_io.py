"""Route-keyed capture-store IO.

The capture is a streaming store directory (``artifacts/capture/<split>/``, see
:mod:`teia.core.capture.store`) written per batch by a stage callback; the eval graph
opens it lazily through :class:`~teia.core.eval.runner.RunEvalSource`. A missing capture is a
hard error — evaluation never re-runs inference. Modality-agnostic: the column layout is the
capture module's concern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from teia.core.capture.store import CaptureStore, CaptureWriter, store_dir
from teia.core.utils import ensure_dir


CAPTURE_SUBDIR = "capture"


def capture_writer(run_dir: str | Path, split: str) -> CaptureWriter:
    """Open the route-keyed capture for writing: one store per run/split, columns named
    ``capture.<route>.<atom>`` / ``batch.<atom>``, modality-agnostic. See
    packages/teia/specs/core/eval.md §4.2."""
    return CaptureWriter(store_dir(run_dir, split, subdir=CAPTURE_SUBDIR), split=split)


def load_capture_store(context: dict[str, Any], split: str = "test") -> CaptureStore:
    """Open the route-keyed capture store, memoized on ``context['report_cache']``."""
    cache = context.setdefault("report_cache", {})
    key = f"{CAPTURE_SUBDIR}:{split}"
    if key not in cache:
        cache[key] = CaptureStore(store_dir(context["run_dir"], split, subdir=CAPTURE_SUBDIR))
    return cache[key]
