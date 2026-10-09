from __future__ import annotations

import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import numpy as np

from teia.core.capture.store import CaptureStore, CaptureWriter, store_dir


class CaptureStoreRoundTripTests(unittest.TestCase):
    def test_numeric_column_round_trips_shape_and_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("scores", shape=(4,), dtype="float32")
            writer.append_numeric("scores", [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
            writer.append_numeric("scores", [[0.9, 1.0, 1.1, 1.2]])
            writer.close()

            store = CaptureStore(root)
            scores = store.numeric("scores")
            self.assertEqual(scores.shape, (3, 4))
            self.assertEqual(scores.dtype, np.dtype("float32"))
            np.testing.assert_allclose(scores[0], [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
            np.testing.assert_allclose(scores[2], [0.9, 1.0, 1.1, 1.2], rtol=1e-6)
            self.assertEqual(store.n_rows, 3)

    def test_integer_dtype_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("targets", shape=(3,), dtype="int8")
            writer.append_numeric("targets", [[1, 0, 1], [0, 0, 1]])
            writer.close()

            targets = CaptureStore(root).numeric("targets")
            self.assertEqual(targets.dtype, np.dtype("int8"))
            np.testing.assert_array_equal(targets, [[1, 0, 1], [0, 0, 1]])

    def test_scalar_row_shape_accepts_single_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("label", shape=(1,), dtype="int64")
            writer.append_numeric("label", [[3], [7]])
            writer.close()

            np.testing.assert_array_equal(CaptureStore(root).numeric("label").ravel(), [3, 7])

    def test_row_shape_mismatch_is_a_hard_error(self) -> None:
        """A capture that silently changed width mid-split would yield a plausible-looking
        but wrong report, so the declaration is enforced on every append."""
        with tempfile.TemporaryDirectory() as tmp:
            writer = CaptureWriter(Path(tmp) / "test", split="test")
            writer.declare_numeric("scores", shape=(4,), dtype="float32")
            with self.assertRaises(ValueError):
                writer.append_numeric("scores", [[0.1, 0.2, 0.3]])

    def test_appending_an_undeclared_column_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CaptureWriter(Path(tmp) / "test", split="test")
            with self.assertRaises(KeyError):
                writer.append_numeric("nope", [[1.0]])

    def test_lane_confusion_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_rows("paths")
            with self.assertRaises(TypeError):
                writer.append_numeric("paths", [[1.0]])
            writer.append_rows("paths", ["a.png"])
            writer.close()
            with self.assertRaises(TypeError):
                CaptureStore(root).numeric("paths")

    def test_duplicate_declaration_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CaptureWriter(Path(tmp) / "test", split="test")
            writer.declare_rows("paths")
            with self.assertRaises(KeyError):
                writer.declare_numeric("paths", shape=(1,))


class RaggedRowLaneTests(unittest.TestCase):
    def test_row_index_seeks_without_reading_neighbours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_rows("instances")
            # deliberately ragged: differing box counts per row
            writer.append_rows(
                "instances",
                [
                    {"boxes": [[0, 0, 1, 1]], "label_indices": [0]},
                    {"boxes": [], "label_indices": []},
                    {"boxes": [[1, 1, 2, 2], [3, 3, 4, 4]], "label_indices": [1, 2]},
                ],
            )
            writer.close()

            rows = CaptureStore(root).rows("instances")
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["label_indices"], [0])
            self.assertEqual(rows[1]["boxes"], [])
            self.assertEqual(len(rows[2]["boxes"]), 2)
            self.assertEqual(rows[-1]["label_indices"], [1, 2])
            self.assertEqual([row["label_indices"] for row in rows], [[0], [], [1, 2]])

    def test_out_of_range_row_raises_indexerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_rows("paths")
            writer.append_rows("paths", ["a.png", "b.png"])
            writer.close()

            rows = CaptureStore(root).rows("paths")
            with self.assertRaises(IndexError):
                rows[2]

    def test_paths_with_unicode_and_commas_survive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            names = ["a,b.png", "ünïcodé/图.png", 'quote".png']
            writer = CaptureWriter(root, split="test")
            writer.declare_rows("paths")
            writer.append_rows("paths", names)
            writer.close()

            self.assertEqual(list(CaptureStore(root).rows("paths")), names)


class BlobLaneTests(unittest.TestCase):
    def test_blob_paths_agree_between_writer_and_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_blob("mask", ext="png")
            path = writer.blob_path("mask", 5)
            path.write_bytes(b"fake")
            writer.record_blob("mask")
            writer.close()

            self.assertEqual(CaptureStore(root).blob_path("mask", 5), path)
            self.assertTrue(CaptureStore(root).blob_path("mask", 5).exists())


class ManifestTests(unittest.TestCase):
    def test_manifest_absence_is_a_hard_error(self) -> None:
        """The manifest is written last, so a partial capture is indistinguishable from a
        missing one — both must fail loudly rather than produce an empty report."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("scores", shape=(2,))
            writer.append_numeric("scores", [[0.1, 0.2]])
            # no close() -> no manifest
            with self.assertRaises(FileNotFoundError):
                CaptureStore(root)

    def test_meta_and_class_names_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="val")
            writer.declare_numeric("scores", shape=(2,))
            writer.append_numeric("scores", [[0.1, 0.2]])
            writer.set_meta(class_names=["a", "b"], task="multi-cls", levels=[0.1, 0.5, 0.9])
            writer.close()

            store = CaptureStore(root)
            self.assertEqual(store.class_names, ["a", "b"])
            self.assertEqual(store.split, "val")
            self.assertEqual(store.meta("task"), "multi-cls")
            self.assertEqual(store.meta("levels"), [0.1, 0.5, 0.9])

    def test_route_slice_resolves_declared_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("capture.multi-cls.scores", shape=(2,), dtype="float32")
            writer.declare_numeric("batch.cls", shape=(2,), dtype="int8")
            writer.append_numeric("capture.multi-cls.scores", [[0.8, 0.2]])
            writer.append_numeric("batch.cls", [[1, 0]])
            writer.declare_route("multi-cls", scores="capture.multi-cls.scores", targets="batch.cls")
            writer.close()

            store = CaptureStore(root)
            self.assertEqual(store.routes(), ["multi-cls"])
            route_slice = store.slice("multi-cls")
            np.testing.assert_allclose(route_slice["scores"][0], [0.8, 0.2], rtol=1e-6)
            np.testing.assert_array_equal(route_slice.get("targets")[0], [1, 0])
            self.assertIsNone(route_slice.get("missing"))

    def test_route_referencing_unknown_column_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = CaptureWriter(Path(tmp) / "test", split="test")
            with self.assertRaises(KeyError):
                writer.declare_route("cls", scores="nope")

    def test_unknown_route_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("s", shape=(1,))
            writer.append_numeric("s", [[1.0]])
            writer.close()
            with self.assertRaises(KeyError):
                CaptureStore(root).slice("det")

    def test_store_dir_layout(self) -> None:
        self.assertEqual(
            store_dir("/runs/r1", "test"), Path("/runs/r1/artifacts/capture/test")
        )


class CaptureMemoryTests(unittest.TestCase):
    def test_writer_peak_memory_is_independent_of_row_count(self) -> None:
        """The O(batch) guarantee. 20k x 256 float32 is ~20MB of data; holding it all in
        Python lists (the pre-store behaviour) cost orders of magnitude more."""
        width, batch, batches = 256, 64, 320
        rows = batch * batches
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("scores", shape=(width,), dtype="float32")
            chunk = np.full((batch, width), 0.5, dtype="float32")

            tracemalloc.start()
            for _ in range(batches):
                writer.append_numeric("scores", chunk)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            writer.close()

            self.assertEqual(CaptureStore(root).numeric("scores").shape, (rows, width))
            # 20MB of data written; peak stays near one batch (64KB) plus interpreter noise.
            self.assertLess(peak, 8 * 1024 * 1024)

    def test_reader_does_not_materialize_the_column(self) -> None:
        rows, width = 20_000, 256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test"
            writer = CaptureWriter(root, split="test")
            writer.declare_numeric("scores", shape=(width,), dtype="float32")
            writer.append_numeric("scores", np.full((rows, width), 0.25, dtype="float32"))
            writer.close()

            tracemalloc.start()
            store = CaptureStore(root)
            scores = store.numeric("scores")
            sampled = float(scores[rows // 2][0])
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            self.assertAlmostEqual(sampled, 0.25, places=6)
            self.assertLess(peak, 4 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
