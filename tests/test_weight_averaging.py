"""TeiaWeightAveraging updates only trainable parameters (frozen backbone stays put)."""
from __future__ import annotations

import unittest

import torch

from teia.core.callbacks import TeiaWeightAveraging


class TeiaWeightAveragingTests(unittest.TestCase):
    def test_multi_avg_fn_skips_frozen_params(self) -> None:
        cb = TeiaWeightAveraging(decay=0.5, tau=1e-9)  # tau→0 makes effective decay ≈ 0.5 immediately
        # Two params: index 0 frozen, index 1 trainable.
        cb._trainable_mask = [False, True]
        averaged = [torch.zeros(3), torch.zeros(3)]
        current = [torch.ones(3), torch.ones(3)]
        cb._multi_avg_fn(averaged, current, torch.tensor(0))
        # Frozen param's average is left untouched...
        self.assertTrue(torch.equal(averaged[0], torch.zeros(3)))
        # ...trainable param's average moved toward `current`.
        self.assertGreater(float(averaged[1].mean()), 0.0)

    def test_multi_avg_fn_falls_back_when_mask_absent(self) -> None:
        cb = TeiaWeightAveraging(decay=0.5, tau=1e-9)
        cb._trainable_mask = None
        averaged = [torch.zeros(2)]
        current = [torch.ones(2)]
        cb._multi_avg_fn(averaged, current, torch.tensor(0))
        self.assertGreater(float(averaged[0].mean()), 0.0)  # all params averaged when mask missing


if __name__ == "__main__":
    unittest.main()
