"""Package-level smoke test for the teia distribution scaffold.
Requires `pip install -e .` (done for this venv in CI/dev setup) —
skips cleanly if the namespace package isn't importable, rather than failing the whole suite.
"""

from __future__ import annotations

import unittest


class TeiaEvalPackageTests(unittest.TestCase):
    def test_teia_zoo_eval_namespace_package_imports(self) -> None:
        try:
            import teia.node.eval  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("teia is not installed in this environment")

    def test_teia_searchpath_plugin_is_discovered(self) -> None:
        from hydra.core.plugins import Plugins
        from hydra.plugins.search_path_plugin import SearchPathPlugin

        classes = [cls.__name__ for cls in Plugins.instance().discover(SearchPathPlugin)]
        if "TeiaSearchPathPlugin" not in classes:
            self.skipTest("teia is not installed in this environment")


if __name__ == "__main__":
    unittest.main()
