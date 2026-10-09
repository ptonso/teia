"""Unit tests for teia.node.eval.view.* — the eval-plane plot/table renderers lifted from
teia.core.report.common.{plots,plot_layout,descriptions,tables}.py.
One test per view class instantiates it with representative constructor kwargs, calls
render() with small synthetic data shaped like the old *_payload() builders took, and asserts
the returned paths exist and are non-empty. A subset of views (RankedBar/ScatterPlot/
MultiCurveBundle/BinaryConfusionGrid/MatrixHeatmap) also gets a pagination test that exceeds the
page-size threshold and asserts multiple files come back.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from teia.node.eval.view.binary_confusion_grid import BINARY_CONFUSION_PAGE_SIZE, BinaryConfusionGrid
from teia.node.eval.view.calibration_diagram import CalibrationDiagram
from teia.node.eval.view.grouped_histogram import GroupedHistogram
from teia.node.eval.view.matrix_heatmap import HEATMAP_TILE_THRESHOLD, MatrixHeatmap
from teia.node.eval.view.multi_curve_bundle import CURVE_PAGE_SIZE, MultiCurveBundle
from teia.node.eval.view.ranked_bar import RANKED_BAR_PAGE_SIZE, RankedBar
from teia.node.eval.view.scatter_plot import SCATTER_PAGE_SIZE, ScatterPlot
from teia.node.eval.view.table import Table
from teia.node.eval.view.threshold_curve import ThresholdCurve


def _assert_nonempty_files(test: unittest.TestCase, paths: list[Path]) -> None:
    test.assertTrue(paths)
    for path in paths:
        test.assertTrue(path.exists(), f"{path} was not written")
        test.assertGreater(path.stat().st_size, 0, f"{path} is empty")


class RankedBarTests(unittest.TestCase):
    def test_render_single_page(self) -> None:
        view = RankedBar(title="Per-Class F1", xlabel="F1", support_label="n")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            written = view.render(out_dir, labels=["cat", "dog", "bird"], values=[0.9, 0.4, 0.7], support=[100, 20, 55])
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0], out_dir / "per_class_f1.png")

    def test_render_vertical_never_paginates(self) -> None:
        view = RankedBar(title="Wide Vertical", orientation="vertical")
        labels = [f"class_{i}" for i in range(50)]
        values = [float(i) for i in range(50)]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), labels=labels, values=values)
            self.assertEqual(len(written), 1)

    def test_render_paginates_beyond_threshold(self) -> None:
        view = RankedBar(title="Dense Bar", description="d", interpretation="i", blindspot="b")
        count = RANKED_BAR_PAGE_SIZE + 15
        labels = [f"class_{i}" for i in range(count)]
        values = [float(i) for i in range(count)]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            written = view.render(out_dir, labels=labels, values=values)
            self.assertGreater(len(written), 1)
            _assert_nonempty_files(self, written)
            for path in written:
                sidecar = out_dir / f"{path.stem}.description.yaml"
                self.assertTrue(sidecar.exists())
                loaded = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
                self.assertEqual(loaded["description"], "d")
                self.assertIn("page", loaded["title"])


class MatrixHeatmapTests(unittest.TestCase):
    def test_render_single_page(self) -> None:
        view = MatrixHeatmap(title="Confusion Matrix")
        matrix = [[0.8, 0.2], [0.1, 0.9]]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), matrix=matrix, x_labels=["a", "b"], y_labels=["a", "b"])
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)

    def test_render_paginates_beyond_tile_threshold(self) -> None:
        view = MatrixHeatmap(title="Dense Confusion")
        n = HEATMAP_TILE_THRESHOLD + 5
        labels = [f"c{i}" for i in range(n)]
        matrix = [[float((row + col) % 3) for col in range(n)] for row in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), matrix=matrix, x_labels=labels, y_labels=labels)
            self.assertGreater(len(written), 1)
            _assert_nonempty_files(self, written)


class ScatterPlotTests(unittest.TestCase):
    def test_render_single_page(self) -> None:
        view = ScatterPlot(title="Precision vs Recall", x_label="Recall", y_label="Precision")
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(
                Path(tmp),
                x=[0.1, 0.5, 0.9],
                y=[0.2, 0.6, 0.8],
                labels=["cat", "dog", "bird"],
                sizes=[50.0, 200.0, 90.0],
            )
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)

    def test_render_paginates_beyond_threshold(self) -> None:
        view = ScatterPlot(title="Dense Scatter", x_label="x", y_label="y")
        count = SCATTER_PAGE_SIZE + 10
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(
                Path(tmp),
                x=[float(i) for i in range(count)],
                y=[float(i % 7) for i in range(count)],
                labels=[f"pt_{i}" for i in range(count)],
            )
            self.assertGreater(len(written), 1)
            _assert_nonempty_files(self, written)


class GroupedHistogramTests(unittest.TestCase):
    def test_render(self) -> None:
        view = GroupedHistogram(title="Confidence Histogram", x_label="Confidence")
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(
                Path(tmp),
                groups={"correct": [0.9, 0.95, 0.99, 0.8], "incorrect": [0.5, 0.4, 0.6]},
            )
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)


class ThresholdCurveTests(unittest.TestCase):
    def test_render(self) -> None:
        view = ThresholdCurve(title="Precision/Recall/F1 vs Confidence", x_label="Threshold", y_label="Score")
        x = [0.1, 0.3, 0.5, 0.7, 0.9]
        series = [
            {"label": "precision", "y": [0.5, 0.6, 0.7, 0.8, 0.9]},
            {"label": "recall", "y": [0.9, 0.8, 0.6, 0.4, 0.2]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), x=x, series=series)
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)


class CalibrationDiagramTests(unittest.TestCase):
    def test_render(self) -> None:
        view = CalibrationDiagram(title="Reliability Diagram")
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(
                Path(tmp),
                mean_confidence=[0.1, 0.3, 0.5, 0.7, 0.9],
                empirical_accuracy=[0.12, 0.28, 0.55, 0.66, 0.91],
                counts=[10, 20, 30, 25, 15],
            )
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)


class MultiCurveBundleTests(unittest.TestCase):
    def test_render_single_page(self) -> None:
        view = MultiCurveBundle(title="PR Curves", x_label="Recall", y_label="Precision", baseline=None)
        curves = [
            {"label": "macro", "x": [0.0, 0.5, 1.0], "y": [1.0, 0.6, 0.1]},
            {"label": "class_a", "x": [0.0, 0.5, 1.0], "y": [1.0, 0.7, 0.2]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), curves=curves)
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)

    def test_render_paginates_beyond_threshold_and_pins_macro(self) -> None:
        view = MultiCurveBundle(title="Dense ROC Curves", x_label="FPR", y_label="TPR", baseline="diagonal")
        curves = [{"label": "macro", "x": [0.0, 1.0], "y": [0.0, 1.0]}]
        count = CURVE_PAGE_SIZE + 8
        curves += [{"label": f"class_{i}", "x": [0.0, 1.0], "y": [0.0, float(i) / count]} for i in range(count)]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), curves=curves)
            self.assertGreater(len(written), 1)
            _assert_nonempty_files(self, written)


class BinaryConfusionGridTests(unittest.TestCase):
    def test_render_single_page(self) -> None:
        view = BinaryConfusionGrid(title="TN-Masked Confusion")
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), labels=["cat", "dog", "bird"], tp=[5.0, 3.0, 2.0], fp=[1.0, 2.0, 0.0], fn=[0.0, 1.0, 3.0])
            _assert_nonempty_files(self, written)
            self.assertEqual(len(written), 1)

    def test_render_paginates_beyond_threshold(self) -> None:
        view = BinaryConfusionGrid(title="Dense Confusion Grid")
        count = BINARY_CONFUSION_PAGE_SIZE + 5
        labels = [f"class_{i}" for i in range(count)]
        tp = [1.0] * count
        fp = [0.5] * count
        fn = [0.2] * count
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), labels=labels, tp=tp, fp=fp, fn=fn)
            self.assertGreater(len(written), 1)
            _assert_nonempty_files(self, written)


class TableTests(unittest.TestCase):
    def test_render_infers_columns(self) -> None:
        view = Table(title="Worst Samples")
        rows = [
            {"sample_id": "s1", "score": 0.1, "filename": "a.png"},
            {"sample_id": "s2", "score": 0.2, "filename": "b.png"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            written = view.render(out_dir, rows=rows)
            _assert_nonempty_files(self, written)
            self.assertEqual(written[0], out_dir / "worst_samples.csv")
            with written[0].open(newline="", encoding="utf-8") as handle:
                loaded_rows = list(csv.DictReader(handle))
            self.assertEqual(loaded_rows[0]["sample_id"], "s1")
            self.assertEqual(loaded_rows[1]["filename"], "b.png")

    def test_render_with_fixed_columns(self) -> None:
        view = Table(title="Fixed Columns Table", columns=["a", "b"])
        rows = [{"a": 1, "b": 2, "c": 3}]
        with tempfile.TemporaryDirectory() as tmp:
            written = view.render(Path(tmp), rows=rows)
            with written[0].open(newline="", encoding="utf-8") as handle:
                loaded_rows = list(csv.DictReader(handle))
            self.assertEqual(set(loaded_rows[0].keys()), {"a", "b"})


class DescriptionSidecarTests(unittest.TestCase):
    def test_no_sidecar_when_no_description_fields_set(self) -> None:
        view = GroupedHistogram(title="No Sidecar", x_label="x")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            view.render(out_dir, groups={"a": [1.0, 2.0]})
            self.assertFalse((out_dir / "no_sidecar.description.yaml").exists())

    def test_sidecar_written_when_description_set(self) -> None:
        view = GroupedHistogram(title="With Sidecar", x_label="x", description="desc text")
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            view.render(out_dir, groups={"a": [1.0, 2.0]})
            sidecar = out_dir / "with_sidecar.description.yaml"
            self.assertTrue(sidecar.exists())
            loaded = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(loaded, {"title": "With Sidecar", "description": "desc text"})


if __name__ == "__main__":
    unittest.main()
