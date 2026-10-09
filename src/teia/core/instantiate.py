from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from teia.core.deps import raise_with_dependency_context


def instantiate(config: Any, **overrides: Any) -> Any:
    from hydra.utils import instantiate as _instantiate
    try:
        return _instantiate(config, **overrides)
    except Exception as exc:
        raise_with_dependency_context(exc, config, feature="Teia object instantiation")


def instantiate_many(config: Any) -> list[Any]:
    from hydra.utils import instantiate as _instantiate
    from omegaconf import DictConfig, ListConfig

    def _is_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))

    try:
        if config is None:
            return []
        if isinstance(config, (dict, DictConfig, Mapping)):
            if "items" in config and isinstance(config["items"], list):
                return [_instantiate(item) for item in config["items"]]
            if "items" in config and isinstance(config["items"], ListConfig):
                return [_instantiate(item) for item in config["items"]]
            if "_target_" in config:
                return [_instantiate(config)]
        if isinstance(config, (list, ListConfig)) or _is_sequence(config):
            return [_instantiate(item) for item in config]
        return [_instantiate(config)]
    except Exception as exc:
        raise_with_dependency_context(exc, config, feature="Teia object instantiation")
