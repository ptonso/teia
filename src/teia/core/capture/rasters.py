"""Blob encoding for ``Blob`` capture atoms, chosen by array dtype and the spec's ``format`` only.

Integer ``[H, W]`` → palette-free mask PNG (uint8/uint16). Float ``[C, H, W]``/``[H, W]`` → RGB PNG.
Float ``[T, C, H, W]`` with ``format: gif`` → an animated GIF. A float image MUST lie in ``[0, 1]``:
core cannot know a tensor's value space, so the producer (an activation kernel, a data transform)
converts to unit range and the encoder raises otherwise. Ground truth and predictions go through the same encoder, so paired metrics
compare like with like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["encode_blob"]

_TOL = 1e-3


def encode_blob(value: Any, path: Path, fmt: str) -> Path:
    from PIL import Image

    array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(array.dtype, np.floating) and array.size and (array.min() < -_TOL or array.max() > 1.0 + _TOL):
        raise ValueError(f"Blob {path}: float image values span [{array.min():.4g}, {array.max():.4g}], expected [0, 1].")
    if fmt == "gif":
        frames = [Image.fromarray(_rgb(frame)) for frame in array]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        return path
    if fmt != "png":
        raise ValueError(f"Blob format {fmt!r} is not supported (png, gif).")
    if np.issubdtype(array.dtype, np.integer) or array.dtype == bool:
        mask = array.squeeze()
        if mask.ndim != 2:
            raise ValueError(f"An integer blob must be a 2D mask, got shape {array.shape}.")
        if mask.size and int(mask.max()) > 65535:
            raise ValueError(f"Mask class index {int(mask.max())} exceeds the uint16 PNG encoding.")
        Image.fromarray(mask.astype("uint16" if mask.size and mask.max() > 255 else "uint8")).save(path)
        return path
    Image.fromarray(_rgb(array)).save(path)
    return path


def _rgb(array: np.ndarray) -> np.ndarray:
    image = array.astype(np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
