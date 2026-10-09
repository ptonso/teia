"""
Multilayer Perceptron.

Source: common knowledge

Description:
  Implements a feed-forward stack of affine layers with optional activation and dropout. These are standard dense-network building blocks rather than a paper-specific reproduction.
"""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "none": nn.Identity,
}


class MLP(TeiaNode):
    """Fully-connected network: Linear → [activation → Linear] × N.

    ``in_shape`` and ``out_shape`` are injected by the pipeline orchestrator
    from the workspace. Both must be 1-D shape tuples, e.g. ``(512,)``.

    Args:
        hidden: Hidden layer widths.
        activation: Activation between hidden layers. One of relu/gelu/silu/tanh/none.
        dropout: Dropout probability applied after each hidden activation (0 = off).
    """

    def build_module(
        self,
        hidden: Sequence[int] = (128,),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        if self.out_shape is None:
            raise ValueError(
                "MLP requires out_shape. Use dict form for out_key: "
                "out_key:\n  pred.X: [dim]"
            )
        in_features = int(self.in_shape[0])
        out_features = int(self.out_shape[0])
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


