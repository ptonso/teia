from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Mapping

CORE_CONFIG_PACKAGE = "teia.core"


@dataclass(frozen=True, slots=True)
class ConfLayer:
    label: str
    uri: str
    path: Path


def _package_conf_dir(config_package: str) -> Path:
    module = importlib.import_module(config_package)
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        return (Path(module_file).parent / "conf").resolve()
    paths = list(getattr(module, "__path__", []))
    if not paths:
        raise ModuleNotFoundError(f"Cannot locate conf directory for package {config_package!r}.")
    return (Path(paths[0]) / "conf").resolve()


def _layer(label: str, path: Path) -> ConfLayer:
    return ConfLayer(label=label, uri=f"file://{path}", path=path)


def core_conf_layer() -> ConfLayer:
    return _layer("core", _package_conf_dir(CORE_CONFIG_PACKAGE))


def plugin_conf_layers() -> list[ConfLayer]:
    """Conf dirs contributed by Hydra search-path plugins (e.g. an installed component library).

    Hydra mounts these at compose time via its own ``SearchPathPlugin`` registry, independent of
    ``resolve_conf_layout``'s primary/project layers. We rediscover them here so anything that
    walks the *real* conf tree outside of an actual Hydra compose call (``teia config
    list/tree/show/explain/lint``) sees the same group inventory Hydra will, without hardcoding
    or naming any specific plugin.
    """
    from hydra.core.plugins import Plugins
    from hydra._internal.config_search_path_impl import ConfigSearchPathImpl
    from hydra.plugins.search_path_plugin import SearchPathPlugin

    search_path = ConfigSearchPathImpl()
    for plugin_cls in Plugins.instance().discover(SearchPathPlugin):
        plugin_cls().manipulate_search_path(search_path)
    layers: list[ConfLayer] = []
    for element in search_path.get_path():
        if not element.path.startswith("file://"):
            continue
        layers.append(ConfLayer(label=element.provider, uri=element.path, path=Path(element.path[len("file://"):])))
    return layers


@dataclass(frozen=True, slots=True)
class ConfLayout:
    primary: ConfLayer
    searchpath: list[ConfLayer]

    @property
    def ordered(self) -> list[ConfLayer]:
        return [self.primary, *self.searchpath]

    @property
    def project(self) -> ConfLayer | None:
        for layer in self.searchpath:
            if layer.label == "project":
                return layer
        return None


def resolve_conf_layout(project_dir: Path, config_dir: Path | None = None) -> ConfLayout:
    primary = core_conf_layer()
    overlay_root = Path(config_dir).resolve() if config_dir is not None else (Path(project_dir).resolve() / "conf")
    searchpath: list[ConfLayer] = []
    if overlay_root.exists():
        searchpath.append(_layer("project", overlay_root))
    searchpath.extend(plugin_conf_layers())
    return ConfLayout(primary=primary, searchpath=searchpath)


def duplicate_paths(layout: ConfLayout) -> dict[str, list[ConfLayer]]:
    """Every config path that more than one layer ships, with the layers that ship it."""
    found: dict[str, list[ConfLayer]] = {}
    for layer in layout.ordered:
        for path in layer.path.rglob("*.yaml"):
            found.setdefault(path.relative_to(layer.path).as_posix(), []).append(layer)
    return {rel: layers for rel, layers in found.items() if len(layers) > 1}


def raise_if_ambiguous(layout: ConfLayout, choices: Mapping[str, Any]) -> None:
    """Fail when an option the run selects exists in more than one layer.

    Hydra would silently take the first layer on its search path, so a project file or a second package at the
    same path would change a run without notice. Only selected options are checked: a clash on an unused option
    is reported by ``teia config lint`` and does not block runs."""
    selected = ["config.yaml"] + [
        f"{key.split('@')[0]}/{value}.yaml"
        for key, values in choices.items()
        if not key.startswith("hydra/")
        for value in (values if isinstance(values, (list, tuple)) else [values])
        if value is not None
    ]
    clashes = {rel: [layer.label for layer in layout.ordered if (layer.path / rel).is_file()] for rel in selected}
    clashes = {rel: labels for rel, labels in clashes.items() if len(labels) > 1}
    if clashes:
        lines = "\n".join(f"  {rel} is shipped by {' and '.join(labels)}" for rel, labels in clashes.items())
        raise ValueError(f"Selected config options exist in more than one layer:\n{lines}\nA config path must be unique across installed packages and the project: rename one copy.")
