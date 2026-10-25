"""Generic frame sink for the live view: ``OpenCvFrameViewer`` blits whatever ``engine.render()`` produced."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


class OpenCvFrameViewer:
    """Show ``[H, W, C]`` (or ``[N, H, W, C]``) RGB frames in an OpenCV window.

    ``cv2`` is imported lazily so headless runs never load it.
    """

    def __init__(self, *, window: str = "teia — live policy", fps: int = 60) -> None:
        self.window = str(window)
        self.fps = int(fps)
        self._cv2: Any = None
        self._delay = max(1, int(1000 / self.fps)) if self.fps > 0 else 1

    def open(self) -> None:
        import os

        # Suppress noisy Qt font warnings triggered by opencv-python's Qt backend
        os.environ["QT_LOGGING_RULES"] = "*=false"

        import cv2

        self._cv2 = cv2
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        # Default to a comfortable ~half monitor size (preserves typical 3:2 / 4:3 Gym aspect ratios well)
        cv2.resizeWindow(self.window, 1200, 800)

    def show(self, frame: Tensor | np.ndarray) -> None:
        cv2 = self._cv2
        if cv2 is None:
            return
        img = self._to_bgr(frame)
        if img is None:
            return
        cv2.imshow(self.window, img)
        cv2.waitKey(self._delay)

    def close(self) -> None:
        if self._cv2 is not None:
            self._cv2.destroyWindow(self.window)
            self._cv2 = None

    def _to_bgr(self, frame: Tensor | np.ndarray) -> np.ndarray | None:
        arr = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return self._cv2.cvtColor(arr, self._cv2.COLOR_RGB2BGR)


__all__ = ["OpenCvFrameViewer"]
