"""
Gated Fusion.

Source: common knowledge

Description:
  Learns a softmax-normalized scalar gate per input from their concatenation, then returns the
  gated weighted sum — a soft, learned alternative to a fixed ``concat.py``/``sum.py`` in the
  fusion family.
"""


from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from teia.base.net import TeiaNode


class GatedFusion(TeiaNode):
    """``in: [feat.<a>, feat.<b>, …]`` (same shape ``[B, D]``)  ``out: feat.<fused>``

    Computes an ``n_inputs``-way softmax gate from the concatenation of all inputs, then returns
    the gate-weighted sum — each input's contribution is learned rather than fixed.
    """

    def build_module(self, **kwargs: Any) -> None:
        shapes = self.in_shape
        self.n_inputs = len(shapes)
        dim = int(shapes[0][-1])
        self.gate = nn.Linear(self.n_inputs * dim, self.n_inputs)

    def forward(self, *feats: Tensor) -> Tensor:
        stacked = torch.stack(feats, dim=1)  # [B, N, D]
        logits = self.gate(torch.cat(feats, dim=-1))  # [B, N]
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, N, 1]
        return (stacked * weights).sum(dim=1)  # [B, D]
