from __future__ import annotations

import unittest
from pathlib import Path

from teia.core.config.lint import run_lint

ROOT = Path(__file__).resolve().parents[1]


class ConfigLintTests(unittest.TestCase):
    def test_full_conf_tree_is_lint_clean(self) -> None:
        findings = run_lint(ROOT)
        self.assertEqual([str(f) for f in findings], [])


class ModuleFolderTests(unittest.TestCase):
    def test_evalmodule_folders_name_a_family_and_other_modules_name_a_task(self) -> None:
        from teia.core.config.lint import check_module_folder

        root = Path("/conf")
        tasks = {"vision-cls"}
        self.assertIsNone(check_module_folder(root / "evalmodule/cls/default.yaml", root, tasks))
        self.assertIsNone(check_module_folder(root / "netmodule/vision-cls/probe.yaml", root, tasks))
        self.assertIsNotNone(check_module_folder(root / "netmodule/cls/probe.yaml", root, tasks))
        self.assertIsNotNone(check_module_folder(root / "evalmodule/default.yaml", root, tasks))
