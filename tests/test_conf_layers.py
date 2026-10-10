"""A config path selected by a run must exist in exactly one layer; lint reports every duplicate."""

from __future__ import annotations

from pathlib import Path

import pytest

from teia.core.config.layers import ConfLayout, _layer, duplicate_paths, raise_if_ambiguous


def _layout(tmp_path: Path) -> ConfLayout:
    for label in ("core", "pkg", "project"):
        (tmp_path / label / "netmodule/x").mkdir(parents=True)
    (tmp_path / "core/config.yaml").write_text("{}\n")
    (tmp_path / "pkg/netmodule/x/a.yaml").write_text("{}\n")
    (tmp_path / "project/netmodule/x/a.yaml").write_text("{}\n")
    (tmp_path / "pkg/netmodule/x/b.yaml").write_text("{}\n")
    return ConfLayout(primary=_layer("core", tmp_path / "core"), searchpath=[_layer("project", tmp_path / "project"), _layer("pkg", tmp_path / "pkg")])


def test_a_selected_option_in_two_layers_fails_naming_both(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="netmodule/x/a.yaml is shipped by project and pkg"):
        raise_if_ambiguous(_layout(tmp_path), {"netmodule": "x/a"})


def test_unique_unselected_and_hydra_choices_pass(tmp_path: Path) -> None:
    raise_if_ambiguous(_layout(tmp_path), {"netmodule": "x/b", "node/net/head@netmodule.head": None, "hydra/launcher": "basic"})


def test_duplicate_paths_lists_every_shared_path(tmp_path: Path) -> None:
    assert {rel: [layer.label for layer in layers] for rel, layers in duplicate_paths(_layout(tmp_path)).items()} == {"netmodule/x/a.yaml": ["project", "pkg"]}
