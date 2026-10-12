"""Media caps apply to frame-stack blobs only; the raster encoder requires float images in [0, 1]."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from teia.base.task import Blob
from teia.core.capture.callback import _AtomWriter
from teia.core.capture.rasters import encode_blob
from teia.core.capture.store import CaptureStore, store_dir
from teia.core.eval.runner import EvalSettings

MEDIA = {"max_items": 2, "max_seconds": 0.3, "max_resolution": 8}


def _writer(tmp: Path) -> _AtomWriter:
    capture = SimpleNamespace(specs={"v.frames": Blob("gif"), "v.image": Blob("png")}, target_specs={})
    return _AtomWriter(tmp, "test", SimpleNamespace(capture_map=capture, contract=None), {}, None, MEDIA)


def test_gif_rows_frames_and_size_are_capped_and_png_is_not(tmp_path):
    writer = _writer(tmp_path)
    for _ in range(2):
        writer.write({"capture.v.frames": [torch.rand(10, 3, 16, 16) for _ in range(3)], "capture.v.image": [torch.rand(3, 16, 16) for _ in range(3)]}, 3)
    writer.close()
    store = CaptureStore(store_dir(tmp_path, "test"))
    gifs, pngs = store.column("capture.v.frames"), store.column("capture.v.image")
    assert len(gifs) == 2 and len(pngs) == 6
    with Image.open(gifs[0]) as gif:
        assert gif.n_frames == 3 and gif.size == (8, 8)


def test_float_outside_unit_range_raises(tmp_path):
    with pytest.raises(ValueError, match="expected \\[0, 1\\]"):
        encode_blob(torch.full((3, 4, 4), -0.5), tmp_path / "a.png", "png")
    encode_blob(torch.full((3, 4, 4), 0.5), tmp_path / "b.png", "png")
    assert np.asarray(Image.open(tmp_path / "b.png")).max() == 128


def test_unknown_media_key_raises():
    with pytest.raises(ValueError, match="unknown keys"):
        EvalSettings(media={"max_frames": 3}).media_caps()
    assert EvalSettings(media={"max_items": 4}).media_caps()["max_resolution"] == 256
