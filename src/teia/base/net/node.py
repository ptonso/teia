"""Neural-network contracts: the base classes every component node implements.

This is the shared, torch-only floor (``teia.base.net``):

* ``TeiaNode`` — shape-aware pipeline node (the orchestrator injects in/out shapes).
* ``ParameterNode`` — a standalone learnable parameter node.
* ``BaseActivation`` — the standardized prediction route (per-sample atom dicts).
* ``BaseLoss`` — a pure backward term.

Concrete implementations (MLP, LinearHead, MSE/CrossEntropy/..., Softmax/Sigmoid/...)
live in ``teia.node.net``; they import their bases from here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, ClassVar

import torch
import torch.nn as nn
from torch import Tensor

Shape = tuple[int, ...]
MultiShape = list[Shape]


class TeiaNode(nn.Module, ABC):
    """Base class for all shape-aware pipeline nodes.

    The orchestrator injects ``in_shape`` and ``out_shape`` at build time so
    nodes can construct weight tensors without declaring dimensions in YAML.

    Shapes are tuples of feature dimensions with NO batch dim.
    e.g. ``(256, 16, 16)`` for a spatial feature map, ``(512,)`` for a vector.

    For multi-input/output nodes, shapes are lists of tuples.

    ``out_shape=None`` is valid for nodes that preserve input topology
    (transformers, normalization layers, etc.).  Such nodes should fall back
    to ``self.in_shape`` inside ``build_module``.
    """

    def __init__(
        self,
        in_shape: Shape | MultiShape,
        out_shape: Shape | MultiShape | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.build_module(**kwargs)

    @abstractmethod
    def build_module(self, **kwargs: Any) -> None:
        """Construct sub-modules using ``self.in_shape`` and ``self.out_shape``."""
        ...

    @abstractmethod
    def forward(self, *args: Any) -> Any:
        ...

    def configure_from_producers(self, producers: dict[str, "TeiaNode"]) -> None:
        """Called once at build, after every node producing one of this node's ``in`` keys is built.

        ``producers`` maps each of this node's ``in`` keys -> the node that produces it.
        ``batch.*`` keys have no producer node and are simply absent from the dict.
        Default: no-op. See teia:core/module/nodes.md.
        """


class ParameterNode(TeiaNode):
    """A standalone learnable parameter emitted as a ``pred.*`` value.

    A fully generic node — a free parameter tensor with no inputs, shaped by ``out_shape``
    (e.g. ``pred.log_alpha: [1]`` for an entropy temperature, a learnable bias/scale, etc.).
    Any positional inputs are ignored (an ``in_key`` may be wired only to fix graph ordering).
    The parameter is trainable and is owned by whichever optimizer the preset routes it to.

    Args:
        init: Constant the parameter is initialized to (default 0.0).
    """

    def build_module(self, init: float = 0.0, **_: Any) -> None:
        shape = _as_shape(self.out_shape)
        self.param = nn.Parameter(torch.full(shape, float(init)))

    def forward(self, *_: Any) -> Tensor:
        return self.param


def _as_shape(out_shape: Any) -> tuple[int, ...]:
    if out_shape is None:
        return (1,)
    if isinstance(out_shape, (list, tuple)) and out_shape and isinstance(out_shape[0], (list, tuple)):
        out_shape = out_shape[0]
    return tuple(int(d) for d in out_shape)


# -- activation contract -------------------------------------------------------


class BaseActivation(nn.Module):
    #: Pure decode kernel collected verbatim into the export bundle. Defined as
    #: ``@staticmethod def kernel(activated, ctx): ...`` on a concrete class.
    kernel: ClassVar[Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None] = None

    #: Explicit opt-in for a route with no kernel (raw activated outputs are the prediction).
    RAW_PASSTHROUGH: bool = False

    def params(self) -> dict[str, Any]:
        """Plain, repr-round-trippable literals this instance's ``kernel`` needs at export
        (e.g. ``QuantileActivation.quantiles``) — the ``BaseActivation`` analog of
        ``IoNode.params()``, since ``kernel`` is a bare staticmethod with no ``self``."""
        return {}

    #: Activated out-atoms whose ONNX axes vary per input, e.g. ``{"boxes": {1: "detections"}}``.
    DYNAMIC_AXES: ClassVar[dict[str, dict[int, str]]] = {}

    def activation(self, *logits: Tensor) -> dict[str, Tensor]:
        """Default: identity over the logit inputs (pure tensor / traceable)."""
        if len(logits) == 1:
            return {"logits": logits[0]}
        return {f"logits_{index}": tensor for index, tensor in enumerate(logits)}

    def postprocess(self, activated: dict[str, Tensor], ctx: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Final: one flat atom dict per sample, from the ``kernel`` staticmethod. Do not override.

        A ``RAW_PASSTHROUGH`` route slices its activated tensors along the batch dim instead."""
        fn = type(self).kernel
        if fn is None:
            if type(self).RAW_PASSTHROUGH:
                rows = len(next(iter(activated.values())))
                return [{key: value[i] for key, value in activated.items()} for i in range(rows)]
            raise RuntimeError(
                f"{type(self).__name__} has no kernel and is not RAW_PASSTHROUGH; "
                "define a kernel staticmethod or set RAW_PASSTHROUGH = True."
            )
        return fn(activated, ctx)

    def configure_from_producers(self, producers: dict[str, Any]) -> None:
        """Called once at build, after every node producing one of this node's ``in`` keys is built.

        ``producers`` maps each of this node's ``in`` keys -> the node that produces it.
        ``batch.*`` keys have no producer node and are simply absent from the dict.
        Default: no-op. See teia:core/module/nodes.md.
        """


class BaseLoss(nn.Module):
    """A pure backward term: ``forward(*in_keys) -> Tensor`` (teia:core/module/activation_route.md).

    Losses own no prediction route — activation/postprocess/decode live in the
    activation node (``BaseActivation``). A composite objective is decomposed into
    atomic loss nodes (e.g. ``recon`` + ``kld`` instead of one composite loss returning a
    tuple); each atomic loss is one ``loss.*`` out_key, weighted and summed by codegen.
    """

    def forward(self, *in_keys: Any) -> Tensor:  # noqa: D401 - documented above
        raise NotImplementedError

    def configure_from_producers(self, producers: dict[str, Any]) -> None:
        """Called once at build, after every node producing one of this node's ``in`` keys is built.

        ``producers`` maps each of this node's ``in`` keys -> the node that produces it.
        ``batch.*`` keys have no producer node and are simply absent from the dict.
        Default: no-op. See teia:core/module/nodes.md.
        """


__all__ = [
    "TeiaNode",
    "Shape",
    "MultiShape",
    "ParameterNode",
    "BaseActivation",
    "BaseLoss",
]
