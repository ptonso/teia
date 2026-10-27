"""
Pixel-space Helpers.

Source: common knowledge

Description:
  Maps between decoder logits, the pixel space a task trains in (``unit`` for ``[0, 1]``, ``signed`` for ``[-1, 1]``) and the ``[0, 1]`` range of exported images. The reconstruction loss and activation share these so they always agree on the space.
"""

from __future__ import annotations

import torch

_EXPECTED = "expected 'unit' or 'signed'."


def squash(logits: torch.Tensor, strategy: str) -> torch.Tensor:
    """Map decoder logits into the pixel space named by ``strategy``."""
    if strategy == "unit":
        return torch.sigmoid(logits)
    if strategy == "signed":
        return torch.tanh(logits)
    raise ValueError(f"unknown pixel_norm strategy {strategy!r}; {_EXPECTED}")


def to_unit(x: torch.Tensor, strategy: str) -> torch.Tensor:
    """Map strategy-space pixels into ``[0, 1]`` for display and export."""
    if strategy == "unit":
        return x
    if strategy == "signed":
        return (x + 1.0) * 0.5
    raise ValueError(f"unknown pixel_norm strategy {strategy!r}; {_EXPECTED}")


def to_strategy(unit: torch.Tensor, strategy: str) -> torch.Tensor:
    """Map ``[0, 1]`` pixels into the configured strategy space."""
    if strategy == "unit":
        return unit
    if strategy == "signed":
        return unit * 2.0 - 1.0
    raise ValueError(f"unknown pixel_norm strategy {strategy!r}; {_EXPECTED}")
