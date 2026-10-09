"""CaptureMap, the capture store writer, and the netmodule step path (teia:core/module/capture_map.md)."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch

from teia.base.task import Blob, Dim, Names, Ragged, TaskContract, Tensor
from teia.core.capture.callback import _AtomWriter
from teia.core.capture.extract import CaptureMap
from teia.core.capture.store import CaptureStore, store_dir
from teia.core.module.net import TeiaNetModule
from teia.node.net.activation.softmax import SoftmaxActivation


class Det(TaskContract):
    meta = {"num_classes": Dim(), "class_names": Names("num_classes")}
    batch = {
        "bboxes": Ragged("N 4", index="batch_idx", target=True),
        "batch_idx": Ragged("N", index="batch_idx", target=True),
        "polys": Ragged("N * 2", index="batch_idx", target=True),
        "mask": Blob("png", target=True),
    }
    capture = {
        "det.boxes": Ragged("N 4", index="det.sample_idx"),
        "det.sample_idx": Ragged("N", index="det.sample_idx"),
        "det.scores": Tensor("B C"),
        "det.mask": Blob("png"),
    }


SOURCES = {"det.boxes": "post.act.boxes", "det.scores": "act.probs", "det.mask": "post.act.mask"}
OUTPUTS = {"act.probs": "act"}
Batch = namedtuple("Batch", ["bboxes", "batch_idx", "polys", "mask"])


def _step(n_per_sample: list[int]) -> tuple[Batch, dict, dict]:
    batch = Batch(
        bboxes=torch.rand(sum(n_per_sample), 4),
        batch_idx=torch.tensor([i for i, n in enumerate(n_per_sample) for _ in range(n)]),
        polys=[[np.random.rand(3 + k, 2) for k in range(n)] for n in n_per_sample],
        mask=torch.zeros(len(n_per_sample), 4, 4, dtype=torch.int64),
    )
    activated = {"act": {"probs": torch.rand(len(n_per_sample), 3)}}
    post = {"act": [{"boxes": np.random.rand(n, 4), "mask": np.eye(4, dtype=np.int64)} for n in n_per_sample]}
    return batch, activated, post


class CaptureMapTests(unittest.TestCase):
    def test_assembles_atoms_by_spec_type(self) -> None:
        batch, activated, post = _step([2, 0, 1])
        atoms = CaptureMap(SOURCES, OUTPUTS, Det()).atoms(batch, activated, post)
        self.assertEqual(tuple(atoms["capture.det.scores"].shape), (3, 3))
        self.assertEqual(tuple(atoms["capture.det.boxes"].shape), (3, 4))
        self.assertEqual(atoms["capture.det.sample_idx"].tolist(), [0, 0, 2])
        self.assertEqual(len(atoms["capture.det.mask"]), 3)
        self.assertEqual(len(atoms["batch.polys"]), 3)  # per-sample polygon lists flattened onto instance rows
        self.assertEqual(len(atoms["batch.mask"]), 3)

    def test_post_aliases_and_bad_sources_fail_fast(self) -> None:
        self.assertEqual(CaptureMap(SOURCES, OUTPUTS, Det()).post_aliases(), {"act"})
        with self.assertRaisesRegex(ValueError, "no activation declares"):
            CaptureMap({"det.scores": "act.logits"}, OUTPUTS, Det())
        with self.assertRaisesRegex(ValueError, "must be an `act.\\*` or `post"):
            CaptureMap({"det.scores": "pred.logits"}, OUTPUTS, Det())
        with self.assertRaisesRegex(ValueError, "not declared by the task contract"):
            CaptureMap({"det.other": "act.probs"}, OUTPUTS, Det())

    def test_bfloat16_is_widened(self) -> None:
        batch, activated, post = _step([1])
        activated["act"]["probs"] = activated["act"]["probs"].to(torch.bfloat16)
        atoms = CaptureMap(SOURCES, OUTPUTS, Det()).atoms(batch, activated, post)
        self.assertEqual(atoms["capture.det.scores"].dtype, torch.float32)


class AtomWriterTests(unittest.TestCase):
    def test_writes_lanes_offsets_indices_and_manifest(self) -> None:
        capture = CaptureMap(SOURCES, OUTPUTS, Det())
        module = SimpleNamespace(capture_map=capture, contract=Det())
        with tempfile.TemporaryDirectory() as tmp:
            writer = _AtomWriter(Path(tmp), "test", module, {"num_classes": 3, "class_names": ["a", "b", "c"]}, None)
            for sizes in ([2, 1], [1]):
                batch, activated, post = _step(sizes)
                writer.write(capture.atoms(batch, activated, post), len(sizes))
            writer.close()
            store = CaptureStore(store_dir(tmp, "test"))
            self.assertEqual(np.asarray(store.column("capture.det.sample_idx")).reshape(-1).tolist(), [0, 0, 1, 2])
            self.assertEqual(np.asarray(store.column("batch.batch_idx")).reshape(-1).tolist(), [0, 0, 1, 2])
            self.assertEqual(len(list(store.column("batch.polys"))), 4)
            self.assertEqual(len(store.column("capture.det.mask")), 3)
            manifest = json.loads((store_dir(tmp, "test") / "manifest.json").read_text())
            self.assertTrue(manifest["task"].endswith("test_capture_map.Det"))
            self.assertEqual(manifest["class_names"], ["a", "b", "c"])
            self.assertIn("boxes", manifest["routes"]["det"])

    def test_max_rows_caps_samples(self) -> None:
        capture = CaptureMap(SOURCES, OUTPUTS, Det())
        module = SimpleNamespace(capture_map=capture, contract=Det())
        with tempfile.TemporaryDirectory() as tmp:
            writer = _AtomWriter(Path(tmp), "test", module, {}, max_rows=2)
            for sizes in ([1, 1], [1, 1]):
                batch, activated, post = _step(sizes)
                writer.write(capture.atoms(batch, activated, post), len(sizes))
            writer.close()
            self.assertEqual(len(CaptureStore(store_dir(tmp, "test")).column("capture.det.scores")), 2)


class NetModuleStepAtomsTests(unittest.TestCase):
    def test_test_step_emits_atoms_through_the_capture_map(self) -> None:
        model = TeiaNetModule()
        model._capture_spec = {"cls.scores": "act.probs", "cls.label": "post.act.label"}
        model._activation_records = [SimpleNamespace(name="act", out_key=["act.probs", "act.label"])]
        model.act = SoftmaxActivation()
        pred_type = namedtuple("Pred", ["logits"])
        model._generated_forward = MethodType(lambda self, batch: pred_type(logits=torch.tensor([[0.1, 2.0], [3.0, 0.0]])), model)
        model._generated_losses = MethodType(lambda self, batch, pred: {"loss": torch.tensor(0.0)}, model)
        model._generated_activation = MethodType(lambda self, pred: {"act": self.act.activation(pred.logits)}, model)
        model.log = MethodType(lambda self, *a, **k: None, model)

        class Cls(TaskContract):
            batch = {"cls": Tensor("B", target=True)}
            capture = {"cls.scores": Tensor("B C"), "cls.label": Tensor("B")}

        model.contract = Cls()
        model.capture_ctx = {}
        batch = namedtuple("Batch", ["cls"])(cls=torch.tensor([1, 0]))
        output = model.test_step(batch, 0)
        self.assertEqual(output["n_samples"], 2)
        self.assertEqual(output["atoms"]["capture.cls.label"].tolist(), [1, 0])
        self.assertEqual(output["atoms"]["batch.cls"].tolist(), [1, 0])
        self.assertEqual(tuple(output["atoms"]["capture.cls.scores"].shape), (2, 2))

    def test_no_capture_ctx_means_no_atoms(self) -> None:
        model = TeiaNetModule()
        model._generated_forward = MethodType(lambda self, batch: None, model)
        model._generated_losses = MethodType(lambda self, batch, pred: {"loss": torch.tensor(0.0)}, model)
        model.log = MethodType(lambda self, *a, **k: None, model)
        self.assertNotIn("atoms", model.test_step(namedtuple("B", ["x"])(x=torch.zeros(2)), 0))


if __name__ == "__main__":
    unittest.main()
