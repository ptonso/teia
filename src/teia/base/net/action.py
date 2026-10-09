"""``teia.base.net.action`` — the action-activation contract for interactive/interactive policies.

Torch-only base sub-protocol: the deterministic ``activation`` route plus the stochastic
``sample`` route. Concrete action activations (gaussian-box, tanh-gaussian, q-argmax, …) live
in ``teia.node.net.activation`` and subclass :class:`ActionActivation`.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

from teia.base.interactive.batch import ActionSample
from teia.base.net.node import BaseActivation

__all__ = ["ActionActivation", "ActionSample"]


class ActionActivation(BaseActivation):
    """Action-space activation: deterministic ``activation`` + stochastic ``sample``.

    The distribution lives here, not in the head — the head emits raw ``pred.*``
    parameters and a distribution-specific activation pairs with a distribution-specific
    loss. ``activation()`` is the export-safe deterministic route; ``sample()`` owns
    stochastic collection (and activation-owned exploration, e.g. epsilon-greedy for
    ``q_argmax``).
    """

    RAW_PASSTHROUGH = True

    @abstractmethod
    def sample(
        self,
        *logits: Tensor,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> ActionSample:
        """Training/evaluation action sampling. Not ONNX-traced.

        Returns an :class:`ActionSample` carrying the distribution-space ``action``,
        the post-scale ``env_action`` sent to ``env.step``, and optional
        ``log_prob`` / ``entropy``. Collection uses ``deterministic=False``;
        eval/infer use ``deterministic=True``.
        """
