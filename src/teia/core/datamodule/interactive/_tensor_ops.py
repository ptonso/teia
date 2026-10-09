"""Tiny tensor helpers shared by the datamodule and the Runner package.

Kept in a leaf module (torch-only, no teia imports) so the interactive modules and the
``runner`` package can import them without forming an import cycle.
"""

from __future__ import annotations

from typing import Any

import torch


def to_device(mapping: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in mapping.items()}


def detach_cpu(value: Any) -> Any:
    if value is None:
        return None
    return value.detach().to("cpu")


def step_rows(item: dict[str, Any], num_envs: int) -> dict[str, torch.Tensor]:
    """The tensor entries of one collected step (leading dim ``num_envs``), detached on the CPU."""
    return {k: detach_cpu(v) for k, v in item.items() if isinstance(v, torch.Tensor) and v.shape[:1] == (num_envs,)}
