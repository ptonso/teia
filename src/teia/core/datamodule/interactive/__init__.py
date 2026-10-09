"""Interactive-regime driver — the core loop variant for online (collect↔train) data generation.

Selected by ``TeiaDataModule`` when the data graph's reader is a live env source (``interactive``).
The regime couples collection to training through a ``Runner`` (sync/async) + a ``PolicyAdapter``
bound to the live policy weights; the env source (``teia.node.data.reader.env``) owns the simulator and
the buffer/collater pillars. See teia:core/datamodule.md (interactive regime).
"""

from __future__ import annotations

from teia.base.interactive.policy import PolicyAdapter
from teia.core.datamodule.interactive.async_runner import AsyncRunner
from teia.core.datamodule.interactive.policy import FeedforwardPolicyAdapter
from teia.core.datamodule.interactive.regime import InteractiveRegime
from teia.core.datamodule.interactive.runner_base import Runner, RunnerContext
from teia.core.datamodule.interactive.sync import SyncRunner

__all__ = [
    "PolicyAdapter",
    "FeedforwardPolicyAdapter",
    "InteractiveRegime",
    "Runner",
    "RunnerContext",
    "SyncRunner",
    "AsyncRunner",
]
