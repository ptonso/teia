"""
Timm Backbone Encoder.

Source:
  - title: "PyTorch Image Models"
    url: "https://github.com/huggingface/pytorch-image-models"
    year: 2019

Description:
  Wraps timm feature backbones as Teia encoders. The module delegates architecture details to timm and exposes selected feature maps to downstream necks or heads.

Adaptations:
  - Uses Teia shape contracts and datamodule-driven configuration around the upstream model.

Taxonomy note (specs/delta/nn-taxonomy-reorg.md Part 4): this node has three modes (global-pool
vector / spatial-pyramid tuple / ViT token sequence), so its fan is mode-dependent — filed here at
its dominant `one_to_one` use. The spatial-pyramid mode is a `one_to_many` use of this
`one_to_one`-filed node; see `one_to_many/grid/README.md`. Not worth splitting unless a second
multi-mode wrapper appears.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from teia.core.deps import require_dependency
from teia.base.net import TeiaNode


class TimmEncoder(TeiaNode):
    """timm backbone over ``(C, H, W)``, in one of three modes.

        pooled  (default):        x -> (D,)
        spatial (out_indices):    x -> one (C_s, H_s, W_s) map per selected stage
        tokens  (tokens=True):    x -> (N, D) patch tokens, prefix tokens dropped

    Args:
        model_name: Any timm model identifier.
        pretrained: Load pretrained weights.
        freeze: Freeze every parameter and keep the model in eval mode.
        global_pool: timm pool for the pooled mode; ``map`` is used instead when the model has an attention pool.
        out_indices: Stages to return; selects the spatial mode.
        tokens: Return patch tokens; selects the tokens mode.
        grad_checkpointing: Recompute each block in backward (timm ``set_grad_checkpointing``); same output, less memory.
        chunk_size: Encode this many images at a time, so memory follows the chunk and not the batch, which holds every node of several boards; under checkpointing in training; pooled and tokens modes only.
    """

    def build_module(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = False,
        global_pool: str = "avg",
        out_indices: Optional[Sequence[int]] = None,
        tokens: bool = False,
        grad_checkpointing: bool = False,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        require_dependency("timm", "TimmEncoder")
        import timm

        self.model_name = model_name
        self.freeze = freeze
        self.chunk_size = chunk_size
        self._spatial = out_indices is not None
        if chunk_size is not None and (chunk_size < 1 or self._spatial):
            raise ValueError(f"TimmEncoder.chunk_size must be a positive int and needs the pooled or token mode; got {chunk_size} with out_indices={out_indices}.")
        self._tokens = tokens
        in_chans = int(self.in_shape[0]) if self.in_shape is not None else 3

        if self._tokens:
            self._model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_chans,
            )
            self._num_prefix = int(getattr(self._model, "num_prefix_tokens", 0))
            self._out_features = self._model.num_features
        elif self._spatial:
            self._model = timm.create_model(
                model_name,
                features_only=True,
                out_indices=tuple(out_indices),
                pretrained=pretrained,
                in_chans=in_chans,
            )
            self._channels = list(self._model.feature_info.channels())
            self._strides = list(self._model.feature_info.reduction())
        else:
            try:
                probe = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="map", in_chans=in_chans)
                resolved_pool = "map" if (hasattr(probe, "attn_pool") and probe.attn_pool is not None) else global_pool
                del probe
            except Exception:
                resolved_pool = global_pool

            self._model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool=resolved_pool,
                in_chans=in_chans,
            )
            self._out_features = self._model.num_features

        if grad_checkpointing:
            self._model.set_grad_checkpointing(True)

        if freeze:
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self._model.eval()
        return self

    def forward(self, img: Tensor) -> "Tensor | tuple[Tensor, ...]":
        if self.chunk_size is None:
            return self._encode(img)
        chunks = img.split(self.chunk_size)
        if self.training and not self.freeze:
            return torch.cat([checkpoint(self._encode, chunk, use_reentrant=False) for chunk in chunks])
        return torch.cat([self._encode(chunk) for chunk in chunks])

    def _encode(self, img: Tensor) -> "Tensor | tuple[Tensor, ...]":
        if self._tokens:
            return self._model.forward_features(img)[:, self._num_prefix:, :]
        if self._spatial:
            tensors = tuple(self._model(img))
            return tensors[0] if len(tensors) == 1 else tensors
        return self._model(img)
