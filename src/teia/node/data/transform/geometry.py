"""Geometry transforms — resize (and, later, letterbox/affine) over the item workspace.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any

import torch

from teia.core.deps import require_dependency
from teia.base.data import Transform


class ResizeImage(Transform):
    """``item.image → item.image`` resized to a square ``image_size`` (uint8 in/out, antialiased)."""

    def __init__(self, *, size: int | tuple[int, int] | None = None) -> None:
        self._size = size
        self.image_size: Any = None  # injected shared field (fallback when ``size`` unset)
        self._resize: Any = None

    def _tuple(self) -> list[int]:
        size = self._size if self._size is not None else self.image_size
        if size is None:
            raise ValueError("ResizeImage needs `size` or an injected `image_size`.")
        return [int(size), int(size)] if isinstance(size, int) else [int(size[0]), int(size[1])]

    def params(self) -> dict[str, Any]:
        return {"size": self._tuple()}

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self._resize is None:
            require_dependency("torchvision", "ResizeImage")
            from torchvision.transforms import v2

            self._resize = v2.Resize(size=self._tuple(), antialias=True)
        return self._resize(image)

    @staticmethod
    def kernel(value: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms import v2

        resize = v2.Resize(size=list(params["size"]), antialias=True)
        return resize(value), {}


class ResizeMask(Transform):
    """``item.mask[H,W] → item.mask[S,S]`` nearest-neighbour (class-index preserving)."""

    def __init__(self, *, size: int | tuple[int, int] | None = None) -> None:
        self._size = size
        self.image_size: Any = None

    def _tuple(self) -> list[int]:
        size = self._size if self._size is not None else self.image_size
        if size is None:
            raise ValueError("ResizeMask needs `size` or an injected `image_size`.")
        return [int(size), int(size)] if isinstance(size, int) else [int(size[0]), int(size[1])]

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        require_dependency("torchvision", "ResizeMask")
        from torchvision.transforms import v2
        from torchvision.transforms.v2 import functional as tvf

        resized = tvf.resize(mask.unsqueeze(0), self._tuple(), interpolation=v2.InterpolationMode.NEAREST)
        return resized.squeeze(0)


def letterbox_boxes(
    boxes: torch.Tensor, r: float, left: int, top: int, h0: int, w0: int, side: int
) -> torch.Tensor:
    """Remap normalized ``cxcywh`` boxes from the original image frame into the letterboxed canvas."""
    if not boxes.numel():
        return boxes
    cx, cy, bw, bh = boxes.unbind(1)
    return torch.stack(
        [(cx * w0 * r + left) / side, (cy * h0 * r + top) / side, bw * w0 * r / side, bh * h0 * r / side],
        dim=1,
    )


class _LetterboxBase(Transform):
    def __init__(self, *, size: int | tuple[int, int] | None = None, fill: int = 114, scaleup: bool = True) -> None:
        self._size = size
        self.fill = int(fill)
        self.scaleup = bool(scaleup)
        self.image_size: Any = None

    def params(self) -> dict[str, Any]:
        return {"size": self._side(), "fill": self.fill, "scaleup": self.scaleup}

    def _side(self) -> int:
        size = self._size if self._size is not None else self.image_size
        if size is None:
            raise ValueError("Letterbox needs `size` or an injected `image_size`.")
        return int(size) if isinstance(size, int) else int(size[0])

    @staticmethod
    def kernel(image: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        from torchvision.transforms.v2 import functional as tvf

        side = int(params["size"])
        fill = int(params.get("fill", 114))
        scaleup = bool(params.get("scaleup", True))
        _, h0, w0 = image.shape
        r = min(side / h0, side / w0)
        if not scaleup:
            r = min(r, 1.0)
        nh, nw = round(h0 * r), round(w0 * r)
        left, top = (side - nw) // 2, (side - nh) // 2
        resized = tvf.resize(image, [nh, nw], antialias=True)
        canvas = tvf.pad(resized, [left, top, side - nw - left, side - nh - top], fill=fill)
        return canvas, {"ratio_pad": ((r, r), (left, top)), "ori_shape": (int(h0), int(w0))}


class Letterbox(_LetterboxBase):
    """Aspect-preserving resize + pad to a square canvas, transforming normalized ``cxcywh`` boxes.

    ``(item.image[3,H0,W0], item.boxes_raw[N,4]) → (item.image[3,S,S], item.boxes[N,4] in the letterboxed
    canvas, item.ratio_pad ``((r,r),(padw,padh))``, item.ori_shape)``.
    """

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor) -> tuple[Any, ...]:
        canvas, restore = type(self).kernel(image, self.params())
        (r, _), (left, top) = restore["ratio_pad"]
        h0, w0 = restore["ori_shape"]
        boxes = letterbox_boxes(boxes, r, left, top, h0, w0, self._side())
        return canvas, boxes, restore["ratio_pad"], restore["ori_shape"]


class LetterboxPose(_LetterboxBase):
    """Letterbox transforming boxes AND normalized keypoints ``[N,K,3]`` (x,y remapped; visibility kept)."""

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor, kpts: torch.Tensor) -> tuple[Any, ...]:
        side = self._side()
        canvas, restore = type(self).kernel(image, self.params())
        (r, _), (left, top) = restore["ratio_pad"]
        h0, w0 = restore["ori_shape"]
        boxes = letterbox_boxes(boxes, r, left, top, h0, w0, side)
        if kpts.numel():
            x = (kpts[..., 0] * w0 * r + left) / side
            y = (kpts[..., 1] * h0 * r + top) / side
            kpts = torch.stack([x, y, kpts[..., 2]], dim=-1)
        return canvas, boxes, kpts, restore["ratio_pad"], restore["ori_shape"]


class LetterboxSeg(_LetterboxBase):
    """Letterbox transforming boxes AND a list of normalized polygons ``list[Tensor[K,2]]``."""

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor, polys: list[torch.Tensor]) -> tuple[Any, ...]:
        side = self._side()
        canvas, restore = type(self).kernel(image, self.params())
        (r, _), (left, top) = restore["ratio_pad"]
        h0, w0 = restore["ori_shape"]
        boxes = letterbox_boxes(boxes, r, left, top, h0, w0, side)
        out_polys = []
        for poly in polys:
            x = (poly[:, 0] * w0 * r + left) / side
            y = (poly[:, 1] * h0 * r + top) / side
            out_polys.append(torch.stack([x, y], dim=1))
        return canvas, boxes, out_polys, restore["ratio_pad"], restore["ori_shape"]
