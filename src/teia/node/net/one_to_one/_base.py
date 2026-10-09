"""
Per-Feature Tokenizer Base.

Source: common knowledge

Description:
  Provides shared shape and concatenation behavior for feature-specific tabular tokenizers. It is a Teia adapter over ordinary tensor projections.
"""


from __future__ import annotations

from typing import Any

from torch import Tensor

from teia.base.net import TeiaNode


def _num_features(in_shape: Any) -> int:
    """Trailing feature count ``N_type`` of a tokenizer ``in_shape``.

    ``(N,)`` for the flat batch, ``(L, N)`` for forecast windowing.
    """
    if in_shape is None:
        return 0
    return int(in_shape[-1])


class PerFeatureTokenizer(TeiaNode):
    """Base: holds ``token_dim`` and the feature count; subclasses build weights."""

    def build_module(self, token_dim: int = 32, **kwargs: Any) -> None:
        self.token_dim = int(token_dim)
        self.n_features = _num_features(self.in_shape)
        self._build_weights(**kwargs)

    def _build_weights(self, **kwargs: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _empty(self, x: Tensor) -> Tensor:
        """Token tensor for a zero-column slot: ``[..., 0, d]``."""
        return x.new_zeros((*x.shape, self.token_dim))
