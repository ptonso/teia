"""Static rules of ``teia plugin check`` over synthetic plugin trees. The environment rules need an installed plugin and are
exercised end to end against a real sandbox project."""

from __future__ import annotations

from pathlib import Path

import pytest

from teia.core import plugin as checker

PYPROJECT = """\
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "myproj"
version = "0.1.0"
license = "Apache-2.0"
dependencies = ["pyteia>=0.1"]

[tool.setuptools.packages.find]
where = ["src", "."]
include = ["myproj*", "hydra_plugins*"]

[tool.setuptools.package-data]
"*" = ["**/*.yaml"]
"""
NODE = '''"""
Conv.

Source: common knowledge

Description:
  A node.
"""

from teia.base.net import TeiaNode


class Conv(TeiaNode):
    """A node."""
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(tmp_path / "pyproject.toml", PYPROJECT)
    _write(tmp_path / "LICENSE", "Apache")
    _write(tmp_path / "hydra_plugins/myproj_searchpath.py", "")
    _write(tmp_path / "src/myproj/node/net/one_to_one/grid/conv.py", NODE)
    _write(tmp_path / "src/myproj/conf/node/net/encoder/myproj/conv.yaml", "_target_: myproj.node.net.one_to_one.grid.conv.Conv\n")
    _write(tmp_path / "src/myproj/conf/netmodule/vision-cls/myproj/conv.yaml", "encoder: {in: batch.image, out: {feat.x: [8]}}\n")
    _write(tmp_path / "src/myproj/conf/task/myproj/vision-cls.yaml", "# @package _global_\ntask: {_target_: teia.task.vision_cls.VisionCls}\n")
    return tmp_path


def _static(root: Path, *, publish: bool = False) -> list[checker.Finding]:
    plugin = checker.load_plugin(root)
    return (
        checker.check_pyproject(plugin)
        + checker.check_license_file(plugin, publish=publish)
        + checker.check_searchpath(plugin)
        + checker.check_targets(plugin)
        + checker.check_docstrings(plugin)
        + checker.check_package(plugin)
        + checker.check_conf_owner(plugin)
    )


def _rules(findings: list[checker.Finding]) -> set[tuple[str, str]]:
    return {(f.level, f.rule) for f in findings}


def test_a_conforming_tree_has_no_static_finding(root: Path) -> None:
    assert _static(root) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: _write(r / "pyproject.toml", PYPROJECT.replace('dependencies = ["pyteia>=0.1"]', "dependencies = []")), ("error", "pyproject")),
        (lambda r: (r / "LICENSE").unlink(), ("warning", "license-file")),
        (lambda r: _write(r / "src/other/x.py", ""), ("error", "package")),
        (lambda r: _write(r / "hydra_plugins/__init__.py", ""), ("error", "searchpath")),
        (lambda r: (r / "hydra_plugins/myproj_searchpath.py").unlink(), ("error", "searchpath")),
        (lambda r: _write(r / "pyproject.toml", PYPROJECT.replace('"*" = ["**/*.yaml"]', '"*" = ["*.txt"]')), ("error", "searchpath")),
        (lambda r: _write(r / "src/myproj/conf/node/net/encoder/conv.yaml", "_target_: x.Y\n"), ("error", "conf-owner")),
        (lambda r: _write(r / "src/myproj/conf/netmodule/vision-cls/conv.yaml", "{}\n"), ("error", "conf-owner")),
        (lambda r: _write(r / "src/myproj/conf/task/vision-cls.yaml", "{}\n"), ("error", "conf-owner")),
        (lambda r: _write(r / "src/myproj/conf/node/net/encoder/myproj/conv.yaml", "_target_: myproj.node.net.one_to_one.grid.conv.Missing\n"), ("error", "target")),
        (lambda r: _write(r / "src/myproj/conf/node/net/encoder/myproj/conv.yaml", "_target_: myproj.node.absent.Conv\n"), ("error", "target")),
        (lambda r: _write(r / "src/myproj/node/net/one_to_one/grid/conv.py", NODE.replace("Source: common knowledge\n", "")), ("error", "docstring")),
        (lambda r: _write(r / "src/myproj/node/net/one_to_one/grid/conv.py", NODE.replace('    """A node."""\n', "    pass\n")), ("error", "docstring")),
    ],
)
def test_each_static_rule_reports_its_violation(root: Path, mutate, expected: tuple[str, str]) -> None:
    mutate(root)
    assert expected in _rules(_static(root))


def test_publish_turns_a_missing_license_into_an_error(root: Path) -> None:
    (root / "LICENSE").unlink()
    assert ("error", "license-file") in _rules(_static(root, publish=True))


@pytest.mark.parametrize(
    ("doc", "problem"),
    [
        ("X.\n\nSource: common knowledge\n\nDescription:\n  X.", None),
        ('X.\n\nSource:\n  - title: "T"\n    url: "U"\n    year: 2020\n\nDescription:\n  X.', None),
        ('X.\n\nSource:\n  - title: "T"\n\nDescription:\n  X.', "title/url/year"),
        ("X.\n\nDescription:\n  X.", "Source: and Description:"),
        ('X.\n\nSource: common knowledge\n\nUpstream:\n  repo: "R"\n  commit: "C"\n  license: "MIT"\n\nDescription:\n  X.', None),
        ('X.\n\nSource: common knowledge\n\nUpstream:\n  repo: "R"\n\nDescription:\n  X.', "commit:, license:"),
    ],
)
def test_provenance_template(doc: str, problem: str | None) -> None:
    found = checker.provenance_problem(doc)
    assert found is None if problem is None else problem in found


def test_component_kind_compares_like_with_like() -> None:
    assert checker.component_kind("myproj.node.net.loss.focal.Focal") == ("net", "loss")
    assert checker.component_kind("teia.node.eval.metric.aligned.x.Y") == ("eval", "metric")


def test_the_shared_namespace_skips_ownership_and_missing_shared_modules(root: Path) -> None:
    _write(root / "src/myproj/conf/node/net/encoder/myproj/conv.yaml", "_target_: myproj.node.elsewhere.Conv\n")
    plugin = checker.load_plugin(root)
    assert checker.check_targets(plugin, shared=True) == []
    assert ("error", "target") in _rules(checker.check_targets(plugin))


def test_scaffold_passes_the_static_rules_and_keeps_existing_files(tmp_path: Path) -> None:
    (tmp_path / "LICENSE").write_text("Apache")
    created = checker.scaffold(tmp_path, "my-proj")
    assert all(made for _, made in created)
    assert _static(tmp_path) == []
    (tmp_path / "pyproject.toml").write_text("kept")
    assert dict(checker.scaffold(tmp_path, "my-proj"))[tmp_path / "pyproject.toml"] is False
    assert (tmp_path / "pyproject.toml").read_text() == "kept"
