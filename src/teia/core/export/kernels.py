"""Tensor-library-agnostic helper shared by activation decode kernels.

Lives in ``teia.core.export`` (not ``teia.base``) because it is export machinery, not a
contract, and must stay collectible without any dependency on ``teia``
(``teia`` has no dependency on either — it's the reverse).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor (CPU/GPU) or array-like to a numpy array.

    Decode kernels run both at infer time (torch tensors, possibly on GPU and in
    reduced precision) and inside the exported bundle (onnxruntime numpy outputs), so they
    normalise through this helper. Floating tensors are upcast to float32 first because
    numpy has no bfloat16/half-on-some-builds support.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.float()
        value = value.numpy()
    return np.asarray(value)
