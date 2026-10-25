"""``ActionSample``: the result of an action activation's ``sample()`` extension."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class ActionSample(NamedTuple):
    """Deliberately omits ``value``: a value is a property of the value head, not of the action distribution, and
    reaches the runner through the capture slice."""

    action: Tensor
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
