"""Tests for the generic, domain-agnostic export pipeline.

The exporter emits a standalone, model-specific runtime package (no teia dependency) by
collecting each activation's decode kernel and each model-input field's preprocess-chain kernels
verbatim. These tests cover the collector, the fail-fast contract, multi-field preprocess
emission, activation-route selection, and the assembler's self-verification — all against fake
fixtures with no modality-specific naming, since the exporter itself has none.
"""
from __future__ import annotations

import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from teia.core.export.assemble import _emit_preprocess, _exec_module, _predictions_equal, write_bundle_runtime
from teia.core.export.bundle import BundleContract, BundlePaths, InputSpec
from teia.core.export.collect import ExportCollectError, collect_callable
from teia.core.export.onnx import _collect_export_artifacts, _disable_mha_fastpath_for_onnx_export, _has_rnn_modules
from teia.core.export.preprocess_slice import PreprocessStep
from teia.core.utils import ensure_dir
from teia.node.net._activation_utils import decode_multilabel
from teia.node.net.activation.sigmoid import SigmoidActivation

Pred = namedtuple("Pred", ["logits"])
_ActRecord = namedtuple("_ActRecord", ["name", "out_key"])


class _MultiClsModule(nn.Module):
    """Minimal stand-in exposing the TeiaNetModule surface the assembler uses."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, num_classes, 1)
        self.act = SigmoidActivation()
        self._activation_records = [_ActRecord(name="act", out_key=["act.probs"])]

    def forward(self, batch):
        return Pred(logits=self.proj(batch.x).mean(dim=(2, 3)))

    def activate(self, pred):
        return {"act": self.act.activation(pred.logits)}

    def postprocess(self, activated, ctx):
        return {"act": self.act.postprocess(activated["act"], ctx)}


def _bad_teia_decode(activated, ctx):
    return ensure_dir(activated["path"])


def _field_a_kernel(value, params):
    return value + params.get("offset", 0.0), {"offset_applied": True}


def _field_b_kernel(value, params):
    return value * params.get("scale", 1.0), {}


def _row_add_one_kernel(value, params):
    return value + 1, {}


def _sum_rows_kernel(values, params):
    return sum(values), {}


class _KernelOwnerFoo:
    @staticmethod
    def kernel(value, params):
        return value + params.get("offset", 0), {}


class _KernelOwnerBar:
    @staticmethod
    def kernel(value, params):
        return value * 2, {}


def _image_contract(num_classes: int, labels: list[str]) -> BundleContract:
    return BundleContract(
        task="multi-cls",
        exporter_family="test",
        inputs=[InputSpec(name="x", payload_key="raw", dtype="float32", shape=["batch", 3, 8, 8])],
        preprocess_chains={"x": [PreprocessStep(kernel=_field_a_kernel, params={"offset": 0.0})]},
        postprocess={
            "task": "multi-cls",
            "activation_outputs": ["probs"],
            "label_names": labels,
            "multi_label_thresholds": [0.5] * num_classes,
            "multi_label_fallback_threshold": 0.5,
        },
        labels=labels,
        outputs=[{"name": "probs", "dtype": "float32", "shape": ["batch", num_classes]}],
    )


class CollectorTests(unittest.TestCase):
    def test_collects_decode_without_teia_imports(self) -> None:
        emitted = collect_callable(decode_multilabel)
        self.assertEqual(emitted.entry, "decode_multilabel")
        rendered = emitted.render()
        self.assertNotIn("teia", rendered)
        self.assertIn("import numpy as np", emitted.imports)
        # transitive kernel deps are inlined verbatim
        self.assertIn("def to_numpy", rendered)
        self.assertIn("def resolve_thresholds", rendered)

    def test_fail_fast_on_teia_core_reference(self) -> None:
        with self.assertRaises(ExportCollectError) as ctx:
            collect_callable(_bad_teia_decode)
        self.assertIn("ensure_dir", str(ctx.exception))

    def test_owner_renames_kernel_entry_to_avoid_collisions(self) -> None:
        """Every class's kernel method is literally named ``kernel``; the owner-based rename is
        what keeps two such methods from colliding in one bundle."""
        foo = collect_callable(_KernelOwnerFoo.kernel, owner="_KernelOwnerFoo")
        bar = collect_callable(_KernelOwnerBar.kernel, owner="_KernelOwnerBar")
        self.assertEqual(foo.entry, "_kernelownerfoo_kernel")
        self.assertEqual(bar.entry, "_kernelownerbar_kernel")
        self.assertIn("def _kernelownerfoo_kernel(", foo.render())
        self.assertNotIn("@staticmethod", foo.render())

    def test_owner_leaves_shared_helper_name_untouched(self) -> None:
        """A differently-named shared helper (e.g. ``decode_multilabel``) is not renamed even
        when an ``owner`` is passed — only a literal ``kernel`` entry is."""
        emitted = collect_callable(decode_multilabel, owner="SomeActivation")
        self.assertEqual(emitted.entry, "decode_multilabel")


class AssembleTests(unittest.TestCase):
    def test_emits_model_specific_teia_free_bundle(self) -> None:
        labels = [f"lbl_{index}" for index in range(4)]
        model = _MultiClsModule(num_classes=4).eval()
        contract = _image_contract(num_classes=4, labels=labels)

        with tempfile.TemporaryDirectory() as tmp:
            paths = BundlePaths.create(Path(tmp))
            # write_bundle_runtime self-verifies (emitted decode == model.postprocess).
            write_bundle_runtime(paths, contract, model)

            postprocess = (paths.runtime_dir / "postprocess.py").read_text(encoding="utf-8")
            preprocess = (paths.runtime_dir / "preprocess.py").read_text(encoding="utf-8")

            # Standalone: nothing from teia leaks into the bundle.
            for text in (postprocess, preprocess):
                self.assertNotIn("import teia", text)
                self.assertNotIn("from teia", text)
            # Specific: this model's decode only, with baked context.
            self.assertIn("def decode_multilabel", postprocess)
            self.assertNotIn("def decode_singlelabel", postprocess)
            self.assertIn("'label_names': ['lbl_0'", postprocess)
            self.assertIn("ROUTES = {'act': ['probs']}", postprocess)
            # Generic glue present.
            self.assertTrue((paths.runtime_dir / "model.py").exists())
            self.assertTrue((paths.runtime_dir / "run.py").exists())

    def test_emitted_decode_matches_loss_postprocess(self) -> None:
        labels = ["a", "b", "c", "d"]
        model = _MultiClsModule(num_classes=4).eval()
        contract = _image_contract(num_classes=4, labels=labels)

        with tempfile.TemporaryDirectory() as tmp:
            paths = BundlePaths.create(Path(tmp))
            write_bundle_runtime(paths, contract, model)
            source = (paths.runtime_dir / "postprocess.py").read_text(encoding="utf-8")

        namespace = _exec_module(source)

        logits = torch.tensor([[-3.0, 3.0, 0.7, -0.2]])
        activated = model.act.activation(logits)
        ctx = {key: value for key, value in contract.postprocess.items() if key != "activation_outputs"}

        bundle_result = namespace["decode"]({"probs": activated["probs"].numpy()})
        reference = model.postprocess({"act": activated}, ctx)
        self.assertTrue(_predictions_equal(bundle_result, reference))
        self.assertEqual(bundle_result["act"][0]["labels"].tolist(), [0, 1, 1, 0])

    def test_node_params_are_baked_per_route_ctx(self) -> None:
        """A route whose activation overrides ``params()`` (the ``QuantileActivation`` pattern —
        a literal a bare ``kernel`` staticmethod can't read off ``self``) gets its own baked ctx
        entry; an unrelated route sharing the bundle's ``CTX_BY_ROUTE`` does not see that key."""

        class _LevelsActivation(nn.Module):
            def params(self):
                return {"levels": [0.1, 0.9]}

            @staticmethod
            def kernel(activated, ctx):
                return {"value": activated["value"].tolist(), "levels": ctx["levels"]}

            def postprocess(self, activated, ctx):
                return type(self).kernel(activated, {**ctx, **self.params()})

        class _LevelsModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = _LevelsActivation()
                self.b = SigmoidActivation()
                self._activation_records = [
                    _ActRecord(name="a", out_key=["act.value"]),
                    _ActRecord(name="b", out_key=["act.probs"]),
                ]

            def forward(self, batch):
                return Pred(logits=batch.x)

            def activate(self, pred):
                return {"a": {"value": pred.logits}, "b": self.b.activation(pred.logits)}

            def postprocess(self, activated, ctx):
                return {
                    "a": self.a.postprocess(activated["a"], ctx),
                    "b": self.b.postprocess(activated["b"], ctx),
                }

        model = _LevelsModule().eval()
        contract = BundleContract(
            task="test",
            exporter_family="test",
            inputs=[InputSpec(name="x", payload_key="raw", dtype="float32", shape=["batch", 2])],
            preprocess_chains={"x": []},
            postprocess={"task": "test", "activation_outputs": ["value", "probs"], "label_names": []},
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = BundlePaths.create(Path(tmp))
            write_bundle_runtime(paths, contract, model)
            postprocess = (paths.runtime_dir / "postprocess.py").read_text(encoding="utf-8")
            namespace = _exec_module(postprocess)

        self.assertEqual(namespace["CTX_BY_ROUTE"]["a"]["levels"], [0.1, 0.9])
        self.assertNotIn("levels", namespace["CTX_BY_ROUTE"]["b"])
        result = namespace["decode"]({"value": torch.tensor([1.0, 2.0]), "probs": torch.tensor([0.1, 0.2])})
        self.assertEqual(result["a"]["levels"], [0.1, 0.9])

    def test_route_subset_excludes_unselected_route_decode(self) -> None:
        """A model with two activation routes; ``routes=["a"]`` must exclude "b" entirely."""

        class _TwoRouteModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.act_a = SigmoidActivation()
                self.act_b = SigmoidActivation()
                self._activation_records = [
                    _ActRecord(name="a", out_key=["act.probs_a"]),
                    _ActRecord(name="b", out_key=["act.probs_b"]),
                ]

            def forward(self, batch):
                return Pred(logits=batch.x)

            def activate(self, pred):
                return {"a": self.act_a.activation(pred.logits), "b": self.act_b.activation(pred.logits)}

            def postprocess(self, activated, ctx):
                return {name: outputs for name, outputs in activated.items()}

        model = _TwoRouteModule().eval()
        contract = BundleContract(
            task="test",
            exporter_family="test",
            inputs=[InputSpec(name="x", payload_key="raw", dtype="float32", shape=["batch", 3])],
            preprocess_chains={"x": []},
            postprocess={"task": "test", "activation_outputs": ["probs_a", "probs_b"], "label_names": []},
            routes=["a"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = BundlePaths.create(Path(tmp))
            write_bundle_runtime(paths, contract, model)
            postprocess = (paths.runtime_dir / "postprocess.py").read_text(encoding="utf-8")
        self.assertIn("'a':", postprocess)
        self.assertNotIn("'b':", postprocess)


class MultiFieldPreprocessTests(unittest.TestCase):
    def test_unrolls_and_merges_independent_field_chains(self) -> None:
        contract = BundleContract(
            task="test",
            exporter_family="test",
            postprocess={},
            inputs=[
                InputSpec(name="a", payload_key="raw_a", dtype="float32", shape=["batch"]),
                InputSpec(name="b", payload_key="raw_b", dtype="float32", shape=["batch"]),
            ],
            preprocess_chains={
                "a": [PreprocessStep(kernel=_field_a_kernel, params={"offset": 1.0})],
                "b": [
                    PreprocessStep(kernel=_field_a_kernel, params={"offset": 2.0}),
                    PreprocessStep(kernel=_field_b_kernel, params={"scale": 3.0}),
                ],
            },
        )
        source = _emit_preprocess(contract)
        # The shared kernel is collected once even though two fields reference it.
        self.assertEqual(source.count("def _field_a_kernel"), 1)

        namespace = _exec_module(source)
        feed, restore = namespace["prepare"]({"raw_a": 5.0, "raw_b": 2.0})
        self.assertEqual(feed["a"], 6.0)
        self.assertEqual(feed["b"], (2.0 + 2.0) * 3.0)
        self.assertEqual(restore, {"offset_applied": True})

    def test_field_with_empty_chain_passes_payload_through(self) -> None:
        contract = BundleContract(
            task="test",
            exporter_family="test",
            postprocess={},
            inputs=[InputSpec(name="a", payload_key="raw_a", dtype="float32", shape=["batch"])],
            preprocess_chains={"a": []},
        )
        namespace = _exec_module(_emit_preprocess(contract))
        feed, restore = namespace["prepare"]({"raw_a": 7.0})
        self.assertEqual(feed["a"], 7.0)
        self.assertEqual(restore, {})

    def test_window_payload_loops_row_mapped_steps_then_reduces_once(self) -> None:
        """A ``"window"``-payload field applies its row_mapped step per row, then its
        collate-seam step once over the whole row list — not the singleton-list form used for a
        ``"value"`` chain."""
        contract = BundleContract(
            task="test",
            exporter_family="test",
            postprocess={},
            inputs=[
                InputSpec(
                    name="x",
                    payload_key="rows",
                    dtype="float32",
                    shape=["batch", "lookback"],
                    payload={"kind": "window"},
                )
            ],
            preprocess_chains={
                "x": [
                    PreprocessStep(kernel=_row_add_one_kernel, params={}, row_mapped=True),
                    PreprocessStep(kernel=_sum_rows_kernel, params={}, row_mapped=False),
                ]
            },
        )
        source = _emit_preprocess(contract)
        namespace = _exec_module(source)
        feed, restore = namespace["prepare"]({"rows": [1, 2, 3]})
        self.assertEqual(feed["x"], 9)  # sum((1+1), (2+1), (3+1))
        self.assertEqual(restore, {})


class ExportArtifactsTests(unittest.TestCase):
    def test_collects_export_artifacts_from_every_graph_node(self) -> None:
        class _NodeWithArtifacts:
            def export_artifacts(self):
                return {"schema": {"columns": ["age"]}}

        class _NodeWithoutArtifacts:
            def export_artifacts(self):
                return {}

        datamodule = SimpleNamespace(
            _live=False,
            _nodes={"resolve_schema": _NodeWithArtifacts(), "numerical": _NodeWithoutArtifacts()},
        )
        self.assertEqual(_collect_export_artifacts(datamodule), {"schema": {"columns": ["age"]}})

    def test_live_stream_graph_has_no_graph_artifacts(self) -> None:
        datamodule = SimpleNamespace(_live=True)
        self.assertEqual(_collect_export_artifacts(datamodule), {})


class OnnxHelperTests(unittest.TestCase):
    """Architecture-shaped ONNX-export quirks (RNN/MHA) — not domain-shaped, so they live here
    rather than in a modality-named test file."""

    def test_disable_mha_fastpath_for_onnx_export_restores_previous_state(self) -> None:
        class _FakeMHA:
            def __init__(self, enabled: bool = True) -> None:
                self.enabled = enabled
                self.calls: list[tuple[str, bool]] = []

            def get_fastpath_enabled(self) -> bool:
                self.calls.append(("get", self.enabled))
                return self.enabled

            def set_fastpath_enabled(self, enabled: bool) -> None:
                self.enabled = bool(enabled)
                self.calls.append(("set", self.enabled))

        mha = _FakeMHA(enabled=True)
        torch_module = SimpleNamespace(backends=SimpleNamespace(mha=mha))

        with _disable_mha_fastpath_for_onnx_export(torch_module):
            self.assertFalse(mha.enabled)

        self.assertTrue(mha.enabled)
        self.assertEqual(mha.calls, [("get", True), ("set", False), ("set", True)])

    def test_has_rnn_modules_detects_rnn(self) -> None:
        model = nn.Sequential(nn.GRU(input_size=3, hidden_size=4, batch_first=True))
        self.assertTrue(_has_rnn_modules(model, torch))

    def test_has_rnn_modules_false_without_rnn(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4))
        self.assertFalse(_has_rnn_modules(model, torch))


if __name__ == "__main__":
    unittest.main()
