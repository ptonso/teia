"""
Soft Q-Critic Node.

Source:
  - title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    url: "https://arxiv.org/abs/1801.01290"
    year: 2018

Description:
  Evaluates concatenated state-action features as a scalar Q-value for soft actor-critic graphs. The node builds a small MLP over `(state, action)` and returns one Q score per batch row.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from teia.base.net import TeiaNode

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class QCritic(TeiaNode):
    """MLP Q-function: ``Q(s, a) = MLP([s, a])``.

    Args:
        hidden: Hidden widths.
        activation: ``relu``, ``tanh``, ``gelu`` or ``silu``.
    """

    def build_module(self, hidden: Any = (256, 256), activation: str = "relu", **_: Any) -> None:
        if activation not in _ACT:
            raise ValueError(f"QCritic.activation must be one of {sorted(_ACT)}; got {activation!r}.")
        state_dim = int(self.in_shape[0][0])
        action_dim = int(self.in_shape[1][0])
        act = _ACT[activation]
        dims = [state_dim + action_dim, *[int(h) for h in hidden]]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[index], dims[index + 1]), act()])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor | None, action: Tensor | None) -> Tensor | None:
        if state is None or action is None:
            return None
        x = torch.cat([state, action.reshape(action.shape[0], -1)], dim=-1)
        return self.net(x)


__all__ = ["QCritic"]
