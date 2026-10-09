from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from teia.core.commands.config import _run_init, _run_show
from teia.core.config.explain import explain_overrides


def _show(selector: str, resolved: bool) -> str:
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(out):
                _run_show(argparse.Namespace(selector=selector, resolved=resolved, overrides=[]))
        finally:
            os.chdir(cwd)
    return out.getvalue()


class ConfigShowTests(unittest.TestCase):
    def test_raw_prints_layer_and_source(self) -> None:
        text = _show("node/net/encoder=mlp_image", resolved=False)
        self.assertIn("# layer: teia", text)
        self.assertIn("_target_:", text)

    def test_resolved_leaf_prints_subtree_at_its_package_address(self) -> None:
        self.assertIn("_target_:", _show("node/net/encoder=mlp_image", resolved=True))

    def test_resolved_task_preset_prints_the_composed_config(self) -> None:
        text = _show("task=vision-cls", resolved=True)
        self.assertIn("netmodule:", text)
        self.assertIn("datamodule:", text)
        self.assertIn("evalmodule:", text)

    def test_missing_option_fails_fast(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _show("node/net/encoder=does_not_exist", resolved=True)


class ConfigInitFromTests(unittest.TestCase):
    def _init(self, tmp: str, value: str) -> None:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _run_init(argparse.Namespace(from_task=value, force=False))
        finally:
            os.chdir(cwd)

    def test_from_task_writes_the_resolved_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._init(tmp, "task=vision-cls")
            for group in ("datamodule", "netmodule", "evalmodule"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, "conf", group, "mine", "vision-cls.yaml")))

    def test_from_requires_the_task_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._init(tmp, "mnist_cls")


class ConfigExplainTests(unittest.TestCase):
    def test_mount_swap_inside_a_task_module_is_attributed_to_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = explain_overrides(Path(tmp), ["task=vision-cls", "node/net/encoder@netmodule.encoder=mlp_tabular"])
        by_group = {group: (option, source) for group, option, source in rows}
        self.assertEqual(by_group["node/net/encoder@netmodule.encoder"], ("mlp_tabular", "CLI"))
        self.assertEqual(by_group["netmodule"], ("vision-cls/mlp", "task"))


if __name__ == "__main__":
    unittest.main()
