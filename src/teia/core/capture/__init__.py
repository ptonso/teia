"""Capture plane: ``CaptureMap`` atoms, the capture callback, the capture store, and walltime.

Evaluation itself (metrics, plots, tables, comparison) is the eval graph — see ``teia.core.eval``
and packages/teia/specs/core/eval.md.
"""

from teia.core.capture.walltime import WalltimeCallback

__all__ = ["WalltimeCallback"]
