"""
Categorical Policy Activation.

Source:
  - title: "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
    url: "https://doi.org/10.1007/BF00992696"
    year: 1992

Description:
  Selects discrete actions from policy logits. Inference uses greedy argmax, while collection samples from a categorical distribution.

Adaptations:
  - Kept as a minimal primitive instead of a full algorithm implementation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from teia.base.net import BaseActivation
from teia.base.interactive.batch import ActionSample


class CategoricalActivation(BaseActivation):
    """Discrete action activation over policy logits ``[B, num_actions]``."""

    RAW_PASSTHROUGH = True  # interactive postprocess is unused; action sampling lives in sample()

    def activation(self, policy: Tensor) -> dict[str, Tensor]:
        """Deterministic greedy action index for export/inference."""
        return {"action": policy.argmax(dim=-1)}

    def sample(
        self,
        *logits: Tensor,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> ActionSample:
        """Greedy eval or Categorical-sampled collection action."""
        policy = logits[0]
        dist = torch.distributions.Categorical(logits=policy)
        if deterministic:
            action = policy.argmax(dim=-1)
        else:
            probs = dist.probs.detach().to("cpu")
            action = torch.multinomial(probs, 1, generator=generator).squeeze(-1).to(policy.device)
        return ActionSample(action=action, env_action=action, log_prob=dist.log_prob(action), entropy=dist.entropy())


__all__ = ["CategoricalActivation"]
