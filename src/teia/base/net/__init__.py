"""``teia.base.net`` — torch-only neural-network contracts.

Node/activation/loss bases live in ``nn.node``; the action-activation contract in ``nn.action``.
Imported directly (``from teia.base.net import TeiaNode, ...``), like ``teia.base.data``/``teia.base.protocols``.
"""

from teia.base.net.action import ActionActivation, ActionSample
from teia.base.net.node import (
    BaseActivation,
    BaseLoss,
    TeiaNode,
    MultiShape,
    ParameterNode,
    Shape,
)

__all__ = [
    "TeiaNode",
    "Shape",
    "MultiShape",
    "ParameterNode",
    "BaseActivation",
    "BaseLoss",
    "ActionActivation",
    "ActionSample",
]
