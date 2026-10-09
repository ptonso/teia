from __future__ import annotations

import unittest
from pathlib import Path

from teia.core.config.lint import run_lint

ROOT = Path(__file__).resolve().parents[1]


class ConfigLintTests(unittest.TestCase):
    def test_full_conf_tree_is_lint_clean(self) -> None:
        findings = run_lint(ROOT)
        self.assertEqual([str(f) for f in findings], [])
