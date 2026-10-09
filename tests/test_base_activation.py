"""Tests for ``BaseActivation.postprocess``'s kernel/RAW_PASSTHROUGH dispatch (final). Covers
the branch directly against the base contract, independent of any concrete node
activation.
"""
from __future__ import annotations

import unittest

from teia.base.net import BaseActivation


class _Kinded(BaseActivation):
    PREDICTION_KIND = "scalar"


class KernelSet(_Kinded):
    @staticmethod
    def kernel(activated, ctx):
        return {"via": "kernel", **ctx}


class RawPassthrough(_Kinded):
    RAW_PASSTHROUGH = True


class NeitherNorPassthrough(_Kinded):
    pass


class BaseActivationPostprocessTests(unittest.TestCase):
    def test_kernel_staticmethod_is_used_when_set(self) -> None:
        result = KernelSet().postprocess({}, {"k": 1})
        self.assertEqual(result, {"via": "kernel", "k": 1})

    def test_raw_passthrough_slices_activated_into_per_sample_atoms(self) -> None:
        activated = {"logits": [10, 20]}
        self.assertEqual(RawPassthrough().postprocess(activated, {}), [{"logits": 10}, {"logits": 20}])

    def test_neither_kernel_nor_passthrough_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "RAW_PASSTHROUGH"):
            NeitherNorPassthrough().postprocess({}, {})


if __name__ == "__main__":
    unittest.main()
