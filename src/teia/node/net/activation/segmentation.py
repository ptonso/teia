"""
Vision Segmentation Softmax Activation.

Source: common knowledge

Description:
  Converts semantic segmentation logits into per-pixel class probabilities. Softmax is applied over the class channel.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from teia.base.net import BaseActivation
from teia.node.net._activation_utils import SquareFrameParams


class SegSoftmaxActivation(SquareFrameParams, BaseActivation):
    """Per-pixel softmax -> argmax mask (ONNX-traceable).

    kernel: resize each predicted mask to its original image shape
    (``ctx['ori_shape']`` per sample, optional).
    """


    def activation(self, logits: Tensor) -> dict[str, Tensor]:
        probs = torch.softmax(logits, dim=1)
        mask = probs.argmax(dim=1).to(torch.int64)
        return {"probs": probs, "mask": mask}

    @staticmethod
    def kernel(activated: dict[str, Tensor], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        masks = torch.as_tensor(activated["mask"]).detach().cpu().to(torch.int64)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        ori_shapes = list(ctx.get("ori_shape") or [])
        predictions: list[dict[str, Any]] = []
        for index, mask in enumerate(masks):
            target = ori_shapes[index] if index < len(ori_shapes) else ctx.get("image_hw")
            if target is not None and tuple(mask.shape[-2:]) != tuple(target):
                mask = (
                    F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=tuple(int(v) for v in target), mode="nearest")
                    .squeeze(0)
                    .squeeze(0)
                    .to(torch.int64)
                )
            predictions.append({"mask": mask.to(torch.uint8)})
        return predictions
