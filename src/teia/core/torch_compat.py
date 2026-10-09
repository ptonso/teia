from __future__ import annotations

from pathlib import Path
from typing import Any


def register_safe_checkpoint_globals(*, torch_module: Any | None = None) -> None:
    """Allow trusted Teia checkpoints to load under PyTorch 2.6+ defaults."""
    try:
        import omegaconf
        if torch_module is None:
            import torch as torch_module
        add_safe_globals = getattr(getattr(torch_module, "serialization", None), "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([
                Any,
                omegaconf.DictConfig,
                omegaconf.ListConfig,
                omegaconf.base.ContainerMetadata,
            ])
    except Exception:
        pass


def load_trusted_checkpoint(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    torch_module: Any | None = None,
) -> Any:
    if torch_module is None:
        import torch as torch_module
    register_safe_checkpoint_globals(torch_module=torch_module)
    return torch_module.load(str(path), map_location=map_location, weights_only=False)
