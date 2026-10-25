"""``teia.core.interactive`` — the stream-mode display + logging support.

Train-loop-adjacent support for stream mode: the live policy-rollout viewer, the online
episodic-metrics logger and the frame sink they use. Distinct from
``teia.core.datamodule.interactive`` (the collection runtime); this branch is display/logging.
Wired by Hydra ``_target_`` (``teia.core.interactive.LiveViewCallback`` / ``OnlineMetricsCallback``).
"""

from teia.core.interactive.live_view import LiveViewCallback
from teia.core.interactive.online import OnlineMetricsCallback
from teia.core.interactive.render import OpenCvFrameViewer

__all__ = [
    "OpenCvFrameViewer",
    "LiveViewCallback",
    "OnlineMetricsCallback",
]
