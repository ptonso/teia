"""
Tabular Token Geometry Helpers.

Source: common knowledge

Description:
  Computes and concatenates token shapes shared by tabular encoders. These helpers define tensor layout rather than a paper-specific model.
"""


from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _shape_list(in_shape: Any) -> list[tuple[int, ...]]:
    if in_shape is None:
        return []
    if isinstance(in_shape, (list, tuple)) and in_shape and isinstance(in_shape[0], (list, tuple)):
        return [tuple(int(v) for v in s) for s in in_shape]
    return [tuple(int(v) for v in in_shape)]


def token_geometry(in_shape: Any) -> tuple[int, int, int | None]:
    """Return ``(n_total, token_dim, lookback)`` from the encoder ``in_shape``.

    Each token shape is ``(N_type, d)`` (flat) or ``(L, N_type, d)`` (windowing).
    ``lookback`` is ``None`` outside windowing mode.
    """
    shapes = _shape_list(in_shape)
    if not shapes:
        return 0, 0, None
    windowed = len(shapes[0]) == 3
    token_dim = int(shapes[0][-1])
    n_total = sum(int(s[-2]) for s in shapes)
    lookback = int(shapes[0][0]) if windowed else None
    return n_total, token_dim, lookback


def concat_tokens(tokens: tuple[Tensor, ...]) -> Tensor:
    """Concatenate per-type token tensors along the feature axis (dim ``-2``)."""
    if len(tokens) == 1:
        return tokens[0]
    return torch.cat(tokens, dim=-2)
