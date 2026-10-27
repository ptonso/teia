"""
Cox Survival Activation.

Source:
  - title: "Regression Models and Life-Tables"
    url: "https://doi.org/10.1111/j.2517-6161.1972.tb00899.x"
    year: 1972

Description:
  Exposes Cox proportional-hazards risk scores as survival predictions. The activation leaves the scalar risk unchanged and formats it for postprocessed outputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from teia.base.net import BaseActivation


class CoxActivation(BaseActivation):
    """Cox proportional-hazards risk ``r = f(x)``, passed through as a ranking score."""


    def activation(self, risk: Tensor) -> dict[str, Tensor]:
        return {"risk": risk.reshape(-1)}

    @staticmethod
    def kernel(activated: dict[str, Tensor], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        del ctx
        risk = torch.as_tensor(activated["risk"]).detach().cpu()
        return [{"risk": float(value)} for value in risk]


__all__ = ["CoxActivation"]
