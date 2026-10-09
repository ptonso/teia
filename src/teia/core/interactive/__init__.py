"""``teia.core.interactive`` — the interactive-regime display + logging support.

Train-loop-adjacent engine support for the interactive regime: the live policy-rollout
viewer, the online episodic-metrics logger, and the render contract they use. Distinct from
``teia.core.datamodule.interactive`` (the collection runtime); this branch is display/logging.
Wired by Hydra ``_target_`` (``teia.core.interactive.LiveViewCallback`` / ``OnlineMetricsCallback``).
"""

from teia.core.interactive.live_view import LiveViewCallback
from teia.core.interactive.online import OnlineMetricsCallback
from teia.core.interactive.render import LiveRenderer, LiveStepRecord, OpenCvFrameViewer

__all__ = [
    "LiveRenderer",
    "LiveStepRecord",
    "OpenCvFrameViewer",
    "LiveViewCallback",
    "OnlineMetricsCallback",
]
