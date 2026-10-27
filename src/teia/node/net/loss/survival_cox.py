"""
Cox Partial-Likelihood Loss.

Source:
  - title: "Regression Models and Life-Tables"
    url: "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x"
    year: 1972

Description:
  Optimizes relative risk scores with the Cox partial likelihood over event times and risk sets. The objective compares each observed event against samples still at risk at that time. Tied times share one risk set (Breslow), so the loss does not depend on the order of the batch.
"""


from __future__ import annotations

import torch
from torch import Tensor

from teia.base.net import BaseLoss


class CoxLoss(BaseLoss):
    """Negative Cox partial log-likelihood (Cox 1972) with Breslow ties.

        L = -Σ_{i: event} (r_i - log Σ_{j: t_j ≥ t_i} exp r_j) / N_events
    """
    def forward(self, risk: Tensor, time: Tensor, event: Tensor) -> Tensor:
        risk = risk.reshape(-1)
        order = torch.argsort(time.reshape(-1), descending=True)
        risk = risk[order]
        time = time.reshape(-1)[order]
        event = event.reshape(-1)[order].float()
        # Sorted descending, the risk set of i is every position up to the last member of i's tie group.
        _, counts = torch.unique_consecutive(time, return_counts=True)
        group_end = counts.cumsum(0) - 1
        last_tied = group_end[torch.repeat_interleave(torch.arange(len(counts), device=time.device), counts)]
        log_risk_set = torch.logcumsumexp(risk, dim=0)[last_tied]
        num_events = event.sum().clamp_min(1.0)
        return -((risk - log_risk_set) * event).sum() / num_events
