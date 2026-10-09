"""Join contract: how/on, present masks (outer), input prefixing, batch_meta."""

from __future__ import annotations

import unittest

import torch

from teia.base.fields import FIELD_VOCAB
from teia.base.data import Collate, Reader, Transform
from teia.core.datamodule import TeiaDataModule
from teia.node.data.join import KeyJoin

_A = {"train": [(0, 1.0), (1, 2.0), (2, 3.0)]}
_B = {"train": [(1, 10.0), (2, 20.0), (3, 30.0)]}


class Src(Reader):
    def __init__(self, data: dict) -> None:
        self._data = data

    def iter_split(self, split: str):
        return self._data.get(split, [])


class Num(Transform):
    def __call__(self, raw):
        return torch.tensor([0.0 if raw is None else raw])


class Stack(Collate):
    field = "numerical"
    structure = "set"

    def __call__(self, field_batch: list):
        return torch.stack(field_batch)


class StackTarget(Stack):
    field = "target"
    structure = None


def _dm(how: str, inputs: tuple[str, ...] = ("a", "b")) -> TeiaDataModule:
    src = {"a": _A, "b": _B}
    nodes = {n: {"_target_": f"{__name__}.Src", "data": src[n], "out": f"stream.{n}"} for n in inputs}
    outs = [f"item.{n}" for n in inputs]
    if how == "outer":
        outs += [f"item.{n}_present" for n in inputs]
    nodes["j"] = {
        "_target_": "teia.node.data.join.KeyJoin", "how": how, "on": "key", "prefix": True,
        "in": [f"stream.{n}" for n in inputs], "out": outs,
    }
    for n in inputs:
        nodes[f"t_{n}"] = {"_target_": f"{__name__}.Num", "phase": "preprocess", "stage": "item", "in": f"item.{n}", "out": f"item.{n}_v"}
        nodes[f"c_{n}"] = {"_target_": f"{__name__}.{'Stack' if n == 'a' else 'StackTarget'}", "in": f"item.{n}_v", "out": f"batch.{'numerical' if n == 'a' else 'target'}"}
    return TeiaDataModule(batch_size=8, num_workers=0, **nodes)


class JoinTest(unittest.TestCase):
    def test_keyjoin_rows(self):
        a, b = dict(_A["train"]), dict(_B["train"])
        self.assertEqual(KeyJoin(how="inner")(a, b), [(2.0, 10.0), (3.0, 20.0)])
        self.assertEqual(KeyJoin(how="left")(a, b), [(1.0, None), (2.0, 10.0), (3.0, 20.0)])
        self.assertEqual(
            KeyJoin(how="outer")(a, b),
            [(1.0, None, True, False), (2.0, 10.0, True, True), (3.0, 20.0, True, True), (None, 30.0, False, True)],
        )

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            KeyJoin(how="cross")
        with self.assertRaises(ValueError):
            KeyJoin(on="id")

    def _batch(self, how):
        dm = _dm(how)
        dm.setup()
        return dm, next(iter(dm.train_dataloader()))

    def test_inner_left_sizes_no_present(self):
        for how, n in (("inner", 2), ("left", 3)):
            dm, batch = self._batch(how)
            self.assertEqual(batch.a_numerical.shape[0], n)
            self.assertFalse(hasattr(batch, "a_present"))
            self.assertEqual(set(dm.batch_meta()), {"a_numerical", "b_target"})

    def test_outer_present_and_meta(self):
        dm, batch = self._batch("outer")
        self.assertEqual(batch.a_present.dtype, torch.bool)
        rows = sorted(zip(batch.a_present.tolist(), batch.b_present.tolist(), batch.b_target.squeeze(1).tolist()))
        self.assertEqual(rows, [(False, True, 30.0), (True, False, 0.0), (True, True, 10.0), (True, True, 20.0)])
        self.assertEqual(
            dm.batch_meta(),
            {"a_numerical": (1,), "b_target": (1,), "a_present": (), "b_present": ()},
        )
        self.assertEqual(dm.batch_type()._fields, ("a_numerical", "b_target", "a_present", "b_present"))
        self.assertIn("a_numerical", FIELD_VOCAB)

    def test_single_input_unprefixed(self):
        dm = _dm("outer", inputs=("a",))
        dm.setup()
        self.assertEqual(dm.batch_meta(), {"numerical": (1,), "a_present": ()})

    def test_bad_out_key_count(self):
        with self.assertRaisesRegex(ValueError, "out_key needs 4"):
            TeiaDataModule(
                a={"_target_": f"{__name__}.Src", "data": _A, "out": "stream.a"},
                b={"_target_": f"{__name__}.Src", "data": _B, "out": "stream.b"},
                j={"_target_": "teia.node.data.join.KeyJoin", "how": "outer", "in": ["stream.a", "stream.b"], "out": ["item.a", "item.b"]},
            )


if __name__ == "__main__":
    unittest.main()
