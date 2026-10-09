"""Unit tests for the data-graph train augments and the executor's augment gating + sibling provider.

Covers the four ported augment nodes (flip / box-hygiene / affine / mosaic) in the normalized item
workspace, plus the executor contract: augment-phase item nodes run train-only and ``needs_siblings``
nodes receive a sibling provider. Replaces the Phase-4-removed datamodule-internal close_mosaic test.
"""

from __future__ import annotations

import unittest

import torch

from teia.base.data import Collate, Reader, Transform
from teia.core.datamodule import TeiaDataModule
from teia.node.data.transform.augment import (
    ClampBoxes,
    Mosaic,
    RandomHFlip,
    RandomPerspectiveCrop,
    SanitizeBoxes,
)


def _wire(node: Transform, in_key: list[str], out_key: list[str]) -> Transform:
    node.in_key, node.out_key = in_key, out_key
    node.phase, node.stage = "augment", "item"
    return node


class AugmentNodeTest(unittest.TestCase):
    def test_hflip_mirrors_image_and_boxes(self) -> None:
        node = _wire(RandomHFlip(p=1.0), ["item.image", "item.boxes"], ["item.image", "item.boxes"])
        image = torch.arange(3 * 4 * 4, dtype=torch.uint8).reshape(3, 4, 4)
        boxes = torch.tensor([[0.25, 0.5, 0.2, 0.4]])
        out_image, out_boxes = node(image, boxes)
        self.assertTrue(torch.equal(out_image, image.flip(-1)))
        self.assertAlmostEqual(float(out_boxes[0, 0]), 0.75, places=5)
        self.assertAlmostEqual(float(out_boxes[0, 2]), 0.2, places=5)

    def test_hflip_reorders_keypoints_by_flip_idx(self) -> None:
        node = _wire(RandomHFlip(p=1.0, flip_idx=[1, 0]), ["item.image", "item.keypoints"],
                     ["item.image", "item.keypoints"])
        image = torch.zeros(3, 4, 4, dtype=torch.uint8)
        kpts = torch.tensor([[[0.1, 0.2, 2.0], [0.8, 0.9, 1.0]]])
        _, out = node(image, kpts)
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0 - 0.8, places=5)
        self.assertEqual(float(out[0, 0, 2]), 1.0)

    def test_sanitize_drops_tiny_boxes_and_syncs_cls(self) -> None:
        node = _wire(SanitizeBoxes(min_size=2.0, min_area=4.0),
                     ["item.image", "item.boxes", "item.box_cls"], ["item.boxes", "item.box_cls"])
        image = torch.zeros(3, 100, 100, dtype=torch.uint8)
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.005, 0.005]])
        cls = torch.tensor([7, 3])
        out_boxes, out_cls = node(image, boxes, cls)
        self.assertEqual(out_boxes.shape[0], 1)
        self.assertEqual(out_cls.tolist(), [7])

    def test_clamp_boxes_clips_to_unit_canvas(self) -> None:
        node = _wire(ClampBoxes(), ["item.boxes"], ["item.boxes"])
        boxes = torch.tensor([[0.9, 0.5, 0.4, 0.2]])
        out = node(boxes)
        x2 = out[0, 0] + out[0, 2] / 2
        self.assertLessEqual(float(x2), 1.0 + 1e-6)

    def test_perspective_crop_outputs_square_and_keeps_norm(self) -> None:
        node = _wire(RandomPerspectiveCrop(size=32, degrees=0.0, translate=(0.0, 0.0), scale=(1.0, 1.0)),
                     ["item.image", "item.boxes", "item.box_cls"],
                     ["item.image", "item.boxes", "item.box_cls"])
        image = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        cls = torch.tensor([2])
        out_image, out_boxes, out_cls = node(image, boxes, cls)
        self.assertEqual(tuple(out_image.shape), (3, 32, 32))
        if out_boxes.numel():
            self.assertGreaterEqual(float(out_boxes.min()), 0.0)
            self.assertLessEqual(float(out_boxes.max()), 1.0)
            self.assertEqual(out_boxes.shape[0], out_cls.shape[0])

    def test_mosaic_composites_to_double_canvas(self) -> None:
        node = _wire(Mosaic(size=16, p=1.0), ["item.image", "item.boxes", "item.box_cls"],
                     ["item.image", "item.boxes", "item.box_cls"])
        base = {
            "item.image": torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8),
            "item.boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "item.box_cls": torch.tensor([1]),
        }
        node.siblings = lambda _i: base
        node.siblings_len = 4
        out_image, out_boxes, out_cls = node(base["item.image"], base["item.boxes"], base["item.box_cls"])
        self.assertEqual(tuple(out_image.shape), (3, 32, 32))
        self.assertEqual(out_boxes.shape[0], out_cls.shape[0])
        self.assertEqual(out_boxes.shape[0], 4)

    def test_mosaic_close_schedule_gate(self) -> None:
        node = Mosaic(size=16, close_mosaic=3)
        node.set_schedule(10)
        node.set_epoch(6)
        self.assertTrue(node.is_open())
        node.set_epoch(7)
        self.assertFalse(node.is_open())


_DATA = {
    "train": [torch.ones(3, 8, 8, dtype=torch.uint8) * i for i in range(4)],
    "val": [torch.ones(3, 8, 8, dtype=torch.uint8) * 9],
    "test": [torch.ones(3, 8, 8, dtype=torch.uint8) * 8],
}


class ImageReader(Reader):
    def iter_split(self, split: str):
        return list(enumerate(_DATA.get(split, [])))


class FlipEverything(Transform):
    """In-test augment: flips the image so train/eval divergence is observable."""

    def __call__(self, image):
        return image.flip(-1)


class SiblingProbe(Transform):
    """In-test cross-sample augment: returns the count of siblings the executor exposed."""

    needs_siblings = True

    def __init__(self) -> None:
        self.siblings = None
        self.siblings_len = 0

    def __call__(self, image):
        return image if self.siblings is None else image + int(self.siblings_len)


class ImageCollate(Collate):
    field = "image"
    structure = "image"

    def __call__(self, field_batch):
        return torch.stack(field_batch)


class AugmentGatingTest(unittest.TestCase):
    def _dm(self, augment_target: str) -> TeiaDataModule:
        return TeiaDataModule(
            batch_size=1,
            num_workers=0,
            rows={"_target_": f"{__name__}.ImageReader", "out": "item.image"},
            aug={
                "_target_": augment_target,
                "in": "item.image",
                "out": "item.image",
                "phase": "augment",
                "stage": "item",
            },
            stack={"_target_": f"{__name__}.ImageCollate", "in": "item.image", "out": "batch.image"},
        )

    def test_augment_runs_train_only(self) -> None:
        dm = self._dm(f"{__name__}.FlipEverything")
        dm.setup()
        seed = dm._seeds_for("train")[0]
        train_ws = dm._item_workspace(seed, augment=True)
        eval_ws = dm._item_workspace(seed, augment=False)
        self.assertTrue(torch.equal(train_ws["item.image"], eval_ws["item.image"].flip(-1)))

    def test_sibling_provider_injected_on_train(self) -> None:
        dm = self._dm(f"{__name__}.SiblingProbe")
        dm.setup()
        dm._bind_siblings("train")
        node = dm._nodes["aug"]
        self.assertIsNotNone(node.siblings)
        self.assertEqual(node.siblings_len, len(_DATA["train"]))
        self.assertIsInstance(node.siblings(0), dict)
        self.assertIn("item.image", node.siblings(0))


if __name__ == "__main__":
    unittest.main()
