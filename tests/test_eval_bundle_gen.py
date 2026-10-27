"""End-to-end test for the ``gen`` evalmodule against a capture-store fixture with real PNG files: the default profile
scores reconstruction FID (``capture.gen.image`` vs ``batch.image``). The Inception extractor is swapped for
``FlattenFeatures`` so the test needs no weights or GPU."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from _store_helpers import load_evalmodule, write_store

from teia.core.eval.graph import build_eval_graph, discover_node_entries
from teia.core.eval.runner import run_eval

FLAT = {"_target_": "teia.node.eval.metric.unpaired.features.FlattenFeatures"}


def _write_png(path: Path, color: tuple[int, int, int], size: int = 8) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, :] = color
    Image.fromarray(array, mode="RGB").save(path)
    return str(path)


class GenBundleEndToEndTests(unittest.TestCase):
    def test_default_profile_scores_reconstruction_fid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = [_write_png(root / "pred" / f"{i}.png", (180, 30 + i, 30)) for i in range(4)]
            real = [_write_png(root / "real" / f"{i}.png", (30, 180 + i, 30)) for i in range(4)]
            write_store(root, "test", {"capture.gen.image": pred, "batch.image": real})
            graph = {name: node for name, node in load_evalmodule("gen").items() if "diagnostics" not in name}
            graph["rfid"] = {**graph["rfid"], "device": "cpu", "feature_extractor": FLAT}

            build_eval_graph(discover_node_entries(graph))
            written = run_eval(root, graph=graph, split="test", out_dir=root / "eval")

            self.assertIn("rfid_table", written)
            summary = yaml.safe_load((root / "eval" / "summary.yaml").read_text())
            self.assertGreater(summary["rfid.frechet_distance"], 0.5)
            self.assertEqual(summary["rfid.n_real"], 4)


if __name__ == "__main__":
    unittest.main()
