from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


from teia.core.commands.config import _render_tree


def _make_tree(root: Path) -> Path:
    group = root / "group"
    group.mkdir(parents=True)
    (group / "leaf.yaml").write_text("_target_: something\n", encoding="utf-8")
    sub = group / "sub"
    sub.mkdir()
    (sub / "inner.yaml").write_text("_target_: something\n", encoding="utf-8")
    return group


def _render(group: Path, **kwargs) -> list[str]:
    lines: list[str] = []
    _render_tree([("test", group)], lines, prefix="", depth=1, **kwargs)
    return lines


class RenderTreeTests(unittest.TestCase):
    def test_default_shows_group_dirs_and_yaml_option_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            group = _make_tree(Path(tmp))
            lines = _render(group, max_level=None, node_only=False)
        joined = "\n".join(lines)
        self.assertIn("sub/", joined)
        self.assertIn("leaf", joined)
        self.assertIn("inner", joined)
        # Only the yaml stem (the option name) is shown, not the file's contents.
        self.assertNotIn(".yaml", joined)
        self.assertNotIn("_target_", joined)

    def test_node_only_drops_leaf_yaml_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            group = _make_tree(Path(tmp))
            lines = _render(group, max_level=None, node_only=True)
        joined = "\n".join(lines)
        self.assertIn("sub/", joined)
        self.assertNotIn("leaf", joined)
        self.assertNotIn("inner", joined)

    def test_level_caps_descent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            group = _make_tree(Path(tmp))
            lines = _render(group, max_level=1, node_only=False)
        joined = "\n".join(lines)
        self.assertIn("sub/", joined)
        self.assertIn("leaf", joined)
        self.assertNotIn("inner", joined)

    def test_last_sibling_uses_corner_connector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            group = _make_tree(Path(tmp))
            lines = _render(group, max_level=1, node_only=False)
        self.assertTrue(lines[-1].startswith("└─ "))
        self.assertTrue(any(line.startswith("├─ ") for line in lines))

    def test_multi_base_entries_are_tagged_by_contributing_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_a = root / "a" / "group"
            group_b = root / "b" / "group"
            group_a.mkdir(parents=True)
            group_b.mkdir(parents=True)
            (group_a / "shared.yaml").write_text("_target_: something\n", encoding="utf-8")
            (group_b / "shared.yaml").write_text("_target_: something\n", encoding="utf-8")
            (group_b / "only_b.yaml").write_text("_target_: something\n", encoding="utf-8")

            lines: list[str] = []
            _render_tree(
                [("core", group_a), ("plugin", group_b)],
                lines,
                prefix="",
                depth=1,
                max_level=None,
                node_only=False,
            )
        joined = "\n".join(lines)
        self.assertIn("shared  [core+plugin]", joined)
        self.assertIn("only_b  [plugin]", joined)


if __name__ == "__main__":
    unittest.main()
