"""Smoke + validation test for the Phase-3 data-graph substrate (teia.base.data + teia.core.datamodule).

Uses tiny in-test components (the test_rl_iterable_io fake-IO pattern) referenced by ``_target_``; core
ships no concrete data components. Covers a 2-node happy path plus the four hard-error cases the executor
must reject at build.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from teia.base.fields import compose_batch
from teia.base.data import Collate, Reader, Reshape, Transform
from teia.core.datamodule import TeiaDataModule
from teia.core.export.preprocess_slice import resolve_data_graph_chain, validate_export_kernels

_DATA = {
    "train": [torch.arange(4.0), torch.arange(4.0) + 1, torch.arange(4.0) + 2],
    "val": [torch.arange(4.0) + 3],
    "test": [torch.arange(4.0) + 4],
}


class LiteralReader(Reader):
    def __init__(self, data: dict | None = None) -> None:
        self._data = data or _DATA

    def iter_split(self, split: str):
        return list(enumerate(self._data.get(split, [])))


class StackCollate(Collate):
    field = "numerical"
    structure = "set"
    kernel = staticmethod(lambda value, params: (value, {}))

    def __call__(self, field_batch: list):
        return torch.stack(field_batch)


class PassThrough(Transform):
    def __call__(self, *inputs):
        return inputs[0]


class KernelRewrite(Transform):
    """A preprocess-phase rewrite that declares a ``kernel``, so it can terminate an
    export chain walk."""

    kernel = staticmethod(lambda value, params: (value, {}))

    def __call__(self, *inputs):
        return inputs[0]


class TargetParse(Transform):
    """A ``target``-phase transform that also supplies an input-lineage key (mirrors
    ``ParseLabelme`` yielding ``item.img_path`` from a LabelMe JSON): it is the data-loading root
    of the input chain, so it declares no ``kernel`` and must terminate the walk."""

    def __call__(self, *inputs):
        return inputs[0]


class SchemaBoundTransform(Transform):
    """Declares the schema contract — the graph must carry a ``plan.schema`` producer."""

    requires_schema = True

    def __call__(self, *inputs):
        return inputs[0]


class MultiInputRewrite(Transform):
    """A rewrite (``out_key == [in_key[0]]``) that also takes an unrelated side input, mirroring
    ``Letterbox``'s ``(item.image, item.boxes_raw) -> item.image`` shape."""

    kernel = staticmethod(lambda value, params: (value, {}))

    def __call__(self, *inputs):
        return inputs[0]


class PassThroughReshape(Reshape):
    def __call__(self, *inputs):
        return inputs[0]


class WindowReshape(Reshape):
    """A reshape node declaring a non-trivial ``payload_spec()``, so its export chain root
    carries the window's shape contract (mirrors ``WindowRows.payload_spec()``)."""

    def payload_spec(self):
        return {"lookback": 4}

    def __call__(self, *inputs):
        return inputs[0]


class GridCollate(Collate):
    """Declares the grid contract — the graph must carry a grid-structured collate."""

    field = "numerical"
    structure = "set"
    requires_structure = "grid"

    def __call__(self, field_batch: list):
        return torch.stack(field_batch)


def _reader(name: str = "rows", out: str = "item.vec") -> dict:
    return {name: {"_target_": f"{__name__}.LiteralReader", "out": out}}


def _collate(in_key: str = "item.vec") -> dict:
    return {"feat": {"_target_": f"{__name__}.StackCollate", "in": in_key, "out": "batch.numerical"}}


def _dm(**nodes) -> TeiaDataModule:
    return TeiaDataModule(batch_size=2, num_workers=0, **nodes)


class IoGraphTest(unittest.TestCase):
    def test_two_node_graph_batch_type_and_meta(self):
        dm = _dm(**_reader(), **_collate())
        dm.setup()
        self.assertIs(dm.batch_type(), compose_batch(["numerical"]))
        self.assertEqual(dm.batch_meta(), {"numerical": (4,)})
        batch = next(iter(dm.train_dataloader()))
        self.assertEqual(type(batch).__name__, "Batch__numerical")
        self.assertEqual(tuple(batch.numerical.shape), (2, 4))

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cycle"):
            _dm(
                a={"_target_": f"{__name__}.PassThrough", "in": "item.b", "out": "item.a", "stage": "item"},
                b={"_target_": f"{__name__}.PassThrough", "in": "item.a", "out": "item.b", "stage": "item"},
            )

    def test_missing_producer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not produced"):
            _dm(**_reader(), **_collate(in_key="item.absent"))

    def test_duplicate_produced_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate out_key"):
            _dm(**_reader("rows"), **_reader("dup"))  # both emit item.vec

    def test_phase_violation_batch_feeding_item_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Phase violation"):
            _dm(
                **_reader(out="item.a"),
                bnode={"_target_": f"{__name__}.PassThrough", "in": "item.a", "out": "item.b", "stage": "batch"},
                cnode={"_target_": f"{__name__}.PassThrough", "in": "item.b", "out": "item.c", "stage": "item"},
            )

    def test_schema_contract_requires_resolve_schema(self):
        with self.assertRaisesRegex(ValueError, "requires a resolved schema"):
            _dm(
                **_reader(out="item.vec"),
                tr={"_target_": f"{__name__}.SchemaBoundTransform", "in": "item.vec", "out": "item.vec2", "stage": "item"},
                feat={"_target_": f"{__name__}.StackCollate", "in": "item.vec2", "out": "batch.numerical"},
            )

    def test_export_chain_skips_augment_phase_rewrite(self):
        """An ``augment``-phase rewrite (e.g. flip) declared after a preprocess rewrite on the
        same key must not block export: the walk steps past it to its predecessor, so the chain
        collects the collate's and the preprocess rewrite's kernels only (2), not the augment
        rewrite's."""
        dm = _dm(
            **_reader(out="item.vec"),
            pre={"_target_": f"{__name__}.KernelRewrite", "in": "item.vec", "out": "item.vec", "stage": "item"},
            aug={
                "_target_": f"{__name__}.PassThrough",
                "in": "item.vec",
                "out": "item.vec",
                "phase": "augment",
                "stage": "item",
            },
            **_collate(),
        )
        dm.setup()
        steps, payload_key, payload = resolve_data_graph_chain(dm, "numerical")
        self.assertEqual(payload_key, "vec")
        self.assertEqual(payload, {"kind": "value"})
        kernels = [step.kernel for step in steps]
        self.assertEqual(len(kernels), 2)
        self.assertIn(KernelRewrite.kernel, kernels)

    def test_export_chain_roots_at_target_phase_transform(self):
        """A ``target``-phase transform that supplies an input-lineage key (like ``ParseLabelme``
        deriving ``item.img_path`` from a JSON alongside its labels) is a data-loading root: the
        walk stops there — collecting only the downstream preprocess and collate kernels (2), not
        the parse itself — and names the field's payload key from the parsed key."""
        dm = _dm(
            **_reader(out="item.json"),
            parse={
                "_target_": f"{__name__}.TargetParse",
                "in": "item.json",
                "out": "item.raw",
                "phase": "target",
                "stage": "item",
            },
            dec={"_target_": f"{__name__}.KernelRewrite", "in": "item.raw", "out": "item.vec", "stage": "item"},
            **_collate(),
        )
        dm.setup()
        steps, payload_key, payload = resolve_data_graph_chain(dm, "numerical")
        self.assertEqual(payload_key, "raw")
        self.assertEqual(payload, {"kind": "value"})
        self.assertEqual(len(steps), 2)
        self.assertIn(KernelRewrite.kernel, [step.kernel for step in steps])

    def test_export_chain_rejects_non_rewrite_augment_node(self):
        """An ``augment``-phase node that is not an in-place rewrite of the key it produces has
        no predecessor to fall back to, so it must still raise."""
        dm = _dm(
            **_reader(out="item.raw"),
            aug={
                "_target_": f"{__name__}.PassThrough",
                "in": "item.raw",
                "out": "item.vec",
                "phase": "augment",
                "stage": "item",
            },
            **_collate(),
        )
        dm.setup()
        with self.assertRaisesRegex(ValueError, "does not rewrite"):
            resolve_data_graph_chain(dm, "numerical")

    def test_export_chain_ignores_unrelated_side_input_on_rewrite(self):
        """A rewrite with an extra, unrelated ``in_key`` (e.g. ``Letterbox``'s ``item.boxes_raw``)
        must not block the walk: it follows the traced key's own rewrite chain and ignores the
        side input's producer entirely, collecting exactly the collate's, the rewrite's, and the
        ``item.raw -> item.vec`` node's kernels (3)."""
        dm = _dm(
            **_reader(out="item.raw"),
            to_vec={"_target_": f"{__name__}.KernelRewrite", "in": "item.raw", "out": "item.vec", "stage": "item"},
            to_side={"_target_": f"{__name__}.PassThrough", "in": "item.raw", "out": "item.side", "stage": "item"},
            rw={
                "_target_": f"{__name__}.MultiInputRewrite",
                "in": ["item.vec", "item.side"],
                "out": "item.vec",
                "stage": "item",
            },
            **_collate(),
        )
        dm.setup()
        steps, payload_key, payload = resolve_data_graph_chain(dm, "numerical")
        self.assertEqual(payload_key, "raw")
        self.assertEqual(payload, {"kind": "value"})
        self.assertEqual(len(steps), 3)

    def test_export_chain_rejects_ambiguous_multi_input_non_rewrite(self):
        """A non-rewrite node with two inputs, neither matching the key it produces, has no
        single upstream key to follow and must still raise."""
        dm = _dm(
            **_reader(out="item.raw"),
            to_a={"_target_": f"{__name__}.PassThrough", "in": "item.raw", "out": "item.a", "stage": "item"},
            to_b={"_target_": f"{__name__}.PassThrough", "in": "item.raw", "out": "item.b", "stage": "item"},
            combine={
                "_target_": f"{__name__}.MultiInputRewrite",
                "in": ["item.a", "item.b"],
                "out": "item.vec",
                "stage": "item",
            },
            **_collate(),
        )
        dm.setup()
        with self.assertRaisesRegex(ValueError, "does not rewrite"):
            resolve_data_graph_chain(dm, "numerical")

    def test_validate_export_kernels_passes_for_fully_covered_chain(self):
        """No exception is the assertion: every node on ``batch.numerical``'s chain has a kernel."""
        dm = _dm(**_reader(out="item.vec"), **_collate())
        dm.setup()
        model = SimpleNamespace(_pipeline_records=[SimpleNamespace(in_key=["batch.numerical"])])
        validate_export_kernels(dm, model)

    def test_validate_export_kernels_raises_on_missing_kernel(self):
        dm = _dm(
            **_reader(out="item.vec"),
            broken={"_target_": f"{__name__}.PassThrough", "in": "item.vec", "out": "item.vec", "stage": "item"},
            **_collate(),
        )
        dm.setup()
        model = SimpleNamespace(_pipeline_records=[SimpleNamespace(in_key=["batch.numerical"])])
        with self.assertRaisesRegex(ValueError, "declares no kernel"):
            validate_export_kernels(dm, model)

    def test_validate_export_kernels_unrelated_reshape_node_does_not_mask_real_problem(self):
        """A ``reshape`` node elsewhere in the graph (not on ``batch.numerical``'s own chain)
        must not suppress validation of the chain that IS actually broken: ``broken`` still
        raises even though an unrelated reshape node exists in the same graph."""
        dm = _dm(
            **_reader(out="item.vec"),
            resh={"_target_": f"{__name__}.PassThroughReshape", "in": "item.vec", "out": "item.reshaped", "stage": "plan"},
            broken={"_target_": f"{__name__}.PassThrough", "in": "item.vec", "out": "item.other", "stage": "item"},
            **_collate(in_key="item.other"),
        )
        dm.setup()
        model = SimpleNamespace(_pipeline_records=[SimpleNamespace(in_key=["batch.numerical"])])
        with self.assertRaisesRegex(ValueError, "declares no kernel"):
            validate_export_kernels(dm, model)

    def test_export_chain_roots_at_reshape_node(self):
        """A ``reshape`` node feeding the collate directly is a windowed chain root: the walk
        terminates there (like ``read``/``join``), the pre-seam transform is marked
        ``row_mapped`` (it loops per row) while the collate seam is not (it runs once over the
        whole row list), and the payload key/spec come from the reshape node's ``payload_spec()``."""
        dm = _dm(
            **_reader(out="item.vec"),
            resh={"_target_": f"{__name__}.WindowReshape", "in": "item.vec", "out": "item.reshaped", "stage": "plan"},
            pre={"_target_": f"{__name__}.KernelRewrite", "in": "item.reshaped", "out": "item.reshaped", "stage": "item"},
            **_collate(in_key="item.reshaped"),
        )
        dm.setup()
        steps, payload_key, payload = resolve_data_graph_chain(dm, "numerical")
        self.assertEqual(payload_key, "reshaped")
        self.assertEqual(payload, {"kind": "window", "lookback": 4})
        self.assertEqual(len(steps), 2)
        self.assertEqual([step.row_mapped for step in steps], [True, False])
        model = SimpleNamespace(_pipeline_records=[SimpleNamespace(in_key=["batch.numerical"])])
        validate_export_kernels(dm, model)

    def test_structure_contract_requires_grid_collate(self):
        with self.assertRaisesRegex(ValueError, "requires 'grid'-structured data"):
            _dm(
                **_reader(out="item.vec"),
                feat={"_target_": f"{__name__}.GridCollate", "in": "item.vec", "out": "batch.numerical"},
            )


if __name__ == "__main__":
    unittest.main()
