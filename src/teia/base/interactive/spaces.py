"""Gymnasium space introspection and numpy<->tensor conversion at the IO boundary.

These helpers translate Gymnasium spaces into the datamodule-owned ``EnvSpec`` and
flatten environment observations into the stable ``obs_<key>`` mapping the collater
turns into ``RlBatch`` fields. They are shared by every environment reader so all agree on the contract.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

# Canonical action-space ids exposed on ``EnvSpec.action_space``.
ACTION_BOX = "box"
ACTION_DISCRETE = "discrete"
ACTION_MULTI_DISCRETE = "multi_discrete"
ACTION_MULTI_BINARY = "multi_binary"


def _np_dtype_name(dtype: Any) -> str:
    return str(np.dtype(dtype).name)


def action_space_fields(space: Any) -> tuple[str, tuple[int, ...], Tensor | None, Tensor | None]:
    """Map a Gymnasium *single* action space to ``EnvSpec`` action fields.

    Returns ``(action_space_id, action_shape, action_low, action_high)``. ``low``/``high``
    are populated only for Box spaces (continuous bounds the activation scales into).
    """
    import gymnasium as gym

    if isinstance(space, gym.spaces.Box):
        shape = tuple(int(d) for d in space.shape)
        low = torch.as_tensor(np.asarray(space.low), dtype=torch.float32)
        high = torch.as_tensor(np.asarray(space.high), dtype=torch.float32)
        return ACTION_BOX, shape, low, high
    if isinstance(space, gym.spaces.Discrete):
        return ACTION_DISCRETE, (int(space.n),), None, None
    if isinstance(space, gym.spaces.MultiDiscrete):
        nvec = tuple(int(n) for n in np.asarray(space.nvec).reshape(-1))
        return ACTION_MULTI_DISCRETE, nvec, None, None
    if isinstance(space, gym.spaces.MultiBinary):
        n = space.n
        shape = (int(n),) if np.isscalar(n) else tuple(int(d) for d in np.asarray(n).reshape(-1))
        return ACTION_MULTI_BINARY, shape, None, None
    raise TypeError(f"Unsupported Gymnasium action space for interactive IO: {type(space).__name__}")


def observation_fields(space: Any) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Map a Gymnasium *single* observation space to ``obs_<key>`` shapes/dtypes.

    A single unkeyed Box becomes ``state``; dict observations contribute one key per
    sub-space (``obs_<key>``). Pixel keys are appended by the IO render path, not here.
    """
    import gymnasium as gym

    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, str] = {}
    if isinstance(space, gym.spaces.Dict):
        for key, sub in space.spaces.items():
            if not isinstance(sub, gym.spaces.Box):
                raise TypeError(
                    f"Dict observation key '{key}' has unsupported space {type(sub).__name__}; "
                    "interactive IO v1 handles Box leaves only."
                )
            shapes[str(key)] = tuple(int(d) for d in sub.shape)
            dtypes[str(key)] = _np_dtype_name(sub.dtype)
        return shapes, dtypes
    if isinstance(space, gym.spaces.Box):
        shapes["state"] = tuple(int(d) for d in space.shape)
        dtypes["state"] = _np_dtype_name(space.dtype)
        return shapes, dtypes
    raise TypeError(f"Unsupported Gymnasium observation space for interactive IO: {type(space).__name__}")


def obs_to_tensors(obs: Any, space: Any) -> dict[str, Tensor]:
    """Flatten a batched vector-env observation into the ``obs_<key>`` tensor mapping.

    The leading axis is ``num_envs``. Dict observations split into one tensor per key;
    a single Box observation is keyed ``state``. Integer obs stay integer; everything
    else is converted faithfully (pixels keep uint8 for the transform to scale).
    """
    import gymnasium as gym

    if isinstance(space, gym.spaces.Dict):
        out: dict[str, Tensor] = {}
        for key in space.spaces:
            out[str(key)] = _to_tensor(obs[key])
        return out
    return {"state": _to_tensor(obs)}


def _to_tensor(array: Any) -> Tensor:
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.floating):
        return torch.as_tensor(arr, dtype=torch.float32)
    if np.issubdtype(arr.dtype, np.integer):
        # uint8 pixels stay uint8 (transform scales); other ints widen to int64.
        if arr.dtype == np.uint8:
            return torch.as_tensor(arr)
        return torch.as_tensor(arr, dtype=torch.int64)
    if np.issubdtype(arr.dtype, np.bool_):
        return torch.as_tensor(arr, dtype=torch.bool)
    return torch.as_tensor(arr.astype(np.float32))


def action_to_numpy(action: Tensor, space: Any) -> np.ndarray:
    """Convert a policy ``env_action`` tensor into the numpy array ``env.step`` expects."""
    import gymnasium as gym

    arr = action.detach().to("cpu").numpy()
    if isinstance(space, (gym.spaces.Discrete,)):
        return arr.astype(np.int64).reshape(-1)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return arr.astype(np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return arr.astype(np.int8)
    # Box: respect the env dtype.
    dtype = space.dtype if getattr(space, "dtype", None) is not None else np.float32
    return arr.astype(dtype)


def scalar_tensor(values: Any, dtype: torch.dtype) -> Tensor:
    """Per-env scalar field (``reward``/``terminated``/``truncated``) as a rank-1 tensor."""
    return torch.as_tensor(np.asarray(values), dtype=dtype).reshape(-1)
