"""``DecodeNodeImages`` — read the image files of a graph's nodes into one tensor.

Source: common knowledge

Description:
  Decodes every path of a node-image list as RGB and stacks the nodes of one graph. With ``size`` each image is
  resized to a square of that side; without it the stored pixels pass through unchanged, so every node image of
  a graph must already share one size. Per-graph and multi-valued, so it has no export kernel.
"""

from __future__ import annotations

import torch

from teia.base.data import Transform
from teia.core.deps import require_dependency


class DecodeNodeImages(Transform):
    """``item.node_image_paths (list[str], N) → item.node_image[N,3,S,S]`` uint8 RGB, ``S`` = ``size`` or the stored side."""

    def __init__(self, *, size: int | None = None) -> None:
        require_dependency("torchvision", "DecodeNodeImages")
        self.size = size

    def __call__(self, paths: list[str]) -> torch.Tensor:
        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms.functional import resize

        images = [read_image(path, mode=ImageReadMode.RGB) for path in paths]
        if self.size is not None:
            images = [resize(image, [self.size, self.size], antialias=True) for image in images]
        shapes = {tuple(image.shape) for image in images}
        if len(shapes) > 1:
            raise ValueError(f"node images of one graph differ in size {sorted(shapes)}; set `size` to resize them.")
        return torch.stack(images) if images else torch.zeros((0, 3, self.size or 0, self.size or 0), dtype=torch.uint8)
