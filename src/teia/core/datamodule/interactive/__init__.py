"""Stream-mode driver — the core loop for engine-sourced (collect↔train) data generation.

Selected by ``TeiaDataModule`` when the data graph carries an ``Engine`` node. A ``Runner`` (sync/async) couples
collection to training through a ``PolicyAdapter`` bound to the live netmodule; the engine owns the simulator and the
buffer (a ``StreamReshape`` node) owns trajectory math. See teia:core/datamodule.md#stream-mode.
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
