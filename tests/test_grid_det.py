"""Grid detection objective and activation: assignment, decode, letterbox restore, NMS and export."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from teia.core.export.collect import collect_callable
from teia.node.net._activation_utils import greedy_nms, grid_postprocess
from teia.node.net.activation.grid_det import GridDetActivation
from teia.node.net.loss.grid_det import GridDetLoss

S, C = 2, 3
BIG = 12.0


def _logit(p):
    p = torch.as_tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def _perfect(boxes, labels, extra=None, kpts=None):
    """Head outputs of one image that match ``boxes`` (fractional cxcywh) exactly, one object per cell."""
    obj = torch.full((1, S, S), -BIG)
    cls = torch.full((1, C, S, S), -BIG)
    box = torch.zeros(1, 4 + (extra == "angle"), S, S)
    kpt = None if kpts is None else torch.zeros(1, *kpts.shape[1:], S, S)
    for i, (cx, cy, w, h, *rest) in enumerate(boxes.tolist()):
        r, c = int(cy * S), int(cx * S)
        obj[0, r, c] = BIG
        cls[0, :, r, c] = -BIG
        cls[0, int(labels[i]), r, c] = BIG
        box[0, :4, r, c] = torch.tensor([cx * S - c, cy * S - r, w, h])
        if extra == "angle":
            box[0, 4, r, c] = rest[0]
        if kpts is not None:
            kpt[0, ..., r, c] = torch.cat([(kpts[i, :, :2] - torch.tensor([cx, cy])) / torch.tensor([w, h]), kpts[i, :, 2:]], dim=-1)
    return (obj, cls, box) if kpt is None else (obj, cls, box, kpt)


BOXES = torch.tensor([[0.25, 0.25, 0.3, 0.2], [0.75, 0.75, 0.2, 0.4]])
LABELS = torch.tensor([1, 2])


def test_perfect_predictions_have_near_zero_loss_and_wrong_ones_do_not():
    heads = _perfect(BOXES, LABELS)
    targets = (BOXES, LABELS, torch.zeros(2, dtype=torch.long))
    assert GridDetLoss()(*heads, *targets) < 1e-2
    wrong = (heads[0], heads[1].roll(1, dims=1), heads[2])
    assert GridDetLoss()(*wrong, *targets) > 1.0


def test_the_larger_box_wins_a_shared_cell_and_no_boxes_give_a_finite_loss():
    boxes = torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.4, 0.4]])
    heads = _perfect(boxes[1:], LABELS[1:])
    assert GridDetLoss()(*heads, boxes, LABELS, torch.zeros(2, dtype=torch.long)) < 1e-2
    empty = GridDetLoss()(*heads, torch.zeros(0, 4), torch.zeros(0), torch.zeros(0))
    assert torch.isfinite(empty) and empty > 1.0


def test_batch_index_selects_the_image():
    obj, cls, box = _perfect(BOXES, LABELS)
    out = GridDetLoss()(obj.repeat(2, 1, 1), cls.repeat(2, 1, 1, 1), box.repeat(2, 1, 1, 1), BOXES.repeat(2, 1), LABELS.repeat(2), torch.tensor([0, 0, 1, 1]))
    assert out < 1e-2


def test_angle_and_keypoints_are_regressed_when_asked():
    rotated = torch.cat([BOXES, torch.tensor([[0.3], [-0.5]])], dim=1)
    heads = _perfect(rotated, LABELS, extra="angle")
    assert GridDetLoss(extra="angle")(*heads, rotated, LABELS, torch.zeros(2, dtype=torch.long)) < 1e-2
    assert GridDetLoss(extra="angle")(heads[0], heads[1], heads[2] + 0.5, rotated, LABELS, torch.zeros(2, dtype=torch.long)) > 1.0
    kpts = torch.tensor([[[0.2, 0.3, 2.0], [0.3, 0.2, 1.0]], [[0.7, 0.8, 2.0], [0.8, 0.7, 0.0]]])
    heads = _perfect(BOXES, LABELS, kpts=kpts)
    obj, cls, box, kpt = heads
    assert GridDetLoss(extra="keypoints")(obj, cls, box, BOXES, LABELS, torch.zeros(2, dtype=torch.long), kpt, kpts) < 1e-2


def test_invalid_extra_raises():
    with pytest.raises(ValueError):
        GridDetLoss(extra="mask")
    with pytest.raises(ValueError):
        GridDetActivation(extra="mask")


def _decode(extra="none", ctx=None, heads=None, **kw):
    act = GridDetActivation(extra=extra)
    act.image_size = 64
    activated = act.activation(*(heads or _perfect(BOXES, LABELS, **kw)))
    return grid_postprocess(activated, {**act.params(), "conf": 0.5, "iou": 0.7, **(ctx or {})})


def test_decode_recovers_the_boxes_scores_and_classes():
    (atoms,) = _decode()
    order = np.argsort(atoms["boxes"][:, 0])
    assert np.allclose(atoms["boxes"][order], BOXES.numpy(), atol=1e-3)
    assert atoms["category"][order].tolist() == [1, 2] and (atoms["score"] > 0.99).all()


def test_decode_restores_the_letterbox_frame():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.1]])
    ctx = {"ori_shape": [(50, 100)], "ratio_pad": [((0.64, 0.64), (0.0, 16.0))]}
    (atoms,) = _decode(ctx=ctx, heads=_perfect(boxes.repeat(1, 1), LABELS[:1]))
    assert np.allclose(atoms["boxes"], [[0.5, 0.5, 0.2, 0.2]], atol=1e-3)


def test_decode_extras_angle_keypoints_and_polygon():
    rotated = torch.cat([BOXES, torch.tensor([[0.3], [-0.5]])], dim=1)
    (atoms,) = _decode("angle", heads=_perfect(rotated, LABELS, extra="angle"))
    order = np.argsort(atoms["boxes"][:, 0])
    assert atoms["boxes"].shape[1] == 5 and np.allclose(atoms["boxes"][order][:, 4], [0.3, -0.5], atol=1e-4)
    kpts = torch.tensor([[[0.2, 0.3, 2.0]], [[0.7, 0.8, 1.0]]])
    (atoms,) = _decode("keypoints", heads=_perfect(BOXES, LABELS, kpts=kpts))
    order = np.argsort(atoms["boxes"][:, 0])
    assert np.allclose(atoms["keypoints"][order], kpts.numpy(), atol=1e-3)
    (atoms,) = _decode("polygon")
    polygon = atoms["polys"][int(np.argsort(atoms["boxes"][:, 0])[0])]
    assert np.allclose(polygon, [[0.1, 0.15], [0.4, 0.15], [0.4, 0.35], [0.1, 0.35]], atol=1e-3)


def test_confidence_and_nms_filter_candidates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=np.float32)
    assert greedy_nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5).tolist() == [0, 2]
    assert greedy_nms(boxes, np.array([0.1, 0.8, 0.7]), 0.5).tolist() == [1, 2]
    assert _decode(ctx={"conf": 0.999999})[0]["boxes"].shape == (0, 4)


def test_kernel_is_export_collectable():
    emitted = collect_callable(GridDetActivation.kernel, owner="GridDetActivation")
    assert emitted.entry == "griddetactivation_kernel" and "teia" not in emitted.render()
