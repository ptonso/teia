"""Image generation task contract.

Source: common knowledge

Description:
  An image in; its reconstruction or sample out, both stored as PNG blobs for distribution metrics. See teia:base/task.md.
"""

from __future__ import annotations

from teia.task._bases import Blob, TaskContract


class VisionGen(TaskContract):
    """``batch.image`` → ``capture.gen.image``; the input image is its own ground truth."""

    batch = {"image": Blob("png", target=True)}
    capture = {"gen.image": Blob("png")}
