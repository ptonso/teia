"""
Node contracts for ``TeiaNetModule``.

Alias-centric module config
---------------------------
Each top-level alias under ``module`` is one instantiated object config.
Hydra imports the implementation YAML into the alias, and Teia reserves the
nested ``node`` mapping for graph metadata:

    module:
      encoder:
        _target_: my_pkg.Encoder
        width: 128
        node:
          in_key: batch.image
          out_key:
            feat.encoded: null

      loss:
        _target_: my_pkg.DetLoss
        node:
          in_key: [pred.raw, batch.boxes]
          weight: 1.0

Public object parameters therefore live at ``module.<alias>.*``, while
Teia-owned orchestration metadata lives at ``module.<alias>.node.*``.
Built-in vision presets standardize on the canonical stage aliases
``encoder``, ``neck``, ``decoder``, ``head``, and ``loss``.

TeiaNode
--------
Subclass ``teia.base.net.TeiaNode`` for all forward graph nodes.  The
orchestrator injects ``in_shape`` and ``out_shape`` at build time
(no batch dim):

    class MyEncoder(TeiaNode):
        def build_module(self, channels=(64, 128), **kwargs):
            in_ch = self.in_shape[0]

        def forward(self, x):
            return self.net(x)

``out_shape=None`` is valid for nodes whose output topology matches their
input.  Such nodes should fall back to ``self.in_shape`` inside
``build_module``.

``node.out_key`` format
-----------------------
``node.out_key`` is always a mapping from workspace key to shape:

    node:
      out_key:
        feat.maps: [256, 16, 16]
        pred.raw: null

Use ``null`` when the shape should be inferred from meta execution.  The
declared key order is preserved for multi-output nodes.

Loss nodes
----------
Loss aliases are plain ``nn.Module`` configs. They are inferred automatically:
any node with at least one ``out_key`` entry starting with ``loss.`` is treated
as a loss node (its ``in_key`` must be ``batch.*`` or ``pred.*`` only).  Their
``forward()`` must accept positional args in the order of ``node.in_key`` and
return a scalar ``Tensor`` (single ``loss.*`` out_key) or a tuple of scalars in
declared order (multiple ``loss.*`` out_keys).

BatchLike
---------
A ``typing.NamedTuple`` whose fields are tensors or Python lists.
Accessed as ``batch.image``, ``batch.cls``, etc.
Must NOT have a ``meta: dict`` field.

Runtime contracts
-----------------
The runtime-checkable ``Protocol`` types below replace the ``getattr``/``hasattr`` duck-typing the
engine and the pipeline module used to probe the datamodule/module surface. They are structural —
a ``LightningDataModule`` subclass satisfies ``TaskDataModule`` by exposing the named members, with
no inheritance burden. ``@runtime_checkable`` only verifies member *presence*; pair with mypy/pyright
for signature checking. ``teia.base`` stays torch-only/import-light, so these use ``typing`` alone.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BatchMetaSource(Protocol):
    """The dry-run / dim-ref surface the pipeline reads off the datamodule.

    ``batch_meta()`` returns ``{field_name: shape}`` where ``shape`` excludes the leading batch
    dim — it seeds the meta workspace and resolves ``out_key`` meta-tokens. ``batch_type()`` returns
    the ``NamedTuple`` batch class instantiated for the example input.
    """

    def batch_meta(self) -> dict[str, tuple[int, ...]]: ...

    def batch_type(self) -> type: ...


@runtime_checkable
class TaskDataModule(BatchMetaSource, Protocol):
    """The dataset-facts contract every Teia datamodule satisfies.

    ``meta()`` returns the facts published at plan time (``num_classes``, ``class_names``, …) that
    dim-refs and the task contract read. ``eval_capture_only`` flags capture-only evaluation: the
    eval step then runs ``forward`` + capture but skips loss generation (RL eval batches are
    forward-valid but carry no loss-only fields such as ``advantage``).
    """

    eval_capture_only: bool

    def meta(self) -> dict[str, Any]: ...


@runtime_checkable
class TeiaStageModule(Protocol):
    """The module surface the engine drives across train/test/predict/export stages.

    ``configure_from_datamodule`` binds task metadata and builds the graph nodes from the
    datamodule's ``batch_meta()``; ``dry_run`` runs a meta forward to lock node shapes.
    """

    def configure_from_datamodule(self, dm: Any) -> None: ...

    def dry_run(self, dm: Any) -> None: ...


@runtime_checkable
class ArtifactPersistable(Protocol):
    """Datamodules that persist fitted state outside the checkpoint (e.g. interactive observation norm).

    ``save_artifacts`` is called after fit; ``load_artifacts`` before test/predict/export, with
    ``required`` raising when a needed artifact is missing.
    """

    def save_artifacts(self, artifacts_dir: Any) -> Any: ...

    def load_artifacts(self, artifacts_dir: Any, *, required: bool) -> Any: ...


@runtime_checkable
class InferSourceBindable(Protocol):
    """Datamodules that accept an infer-time ``--src`` override (re-point the predict split)."""

    def bind_infer_source(self, src: Any) -> None: ...


@runtime_checkable
class TaskMetadataReady(Protocol):
    """Datamodules that lazily resolve task metadata (class names, shapes) on demand."""

    def ensure_task_metadata(self) -> None: ...


@runtime_checkable
class NativeStageRunner(Protocol):
    """Modules that own their full stage execution, bypassing the Lightning ``Trainer``.

    Optional: when a module implements ``run_teia_stage`` the engine hands it the whole stage; an
    ``AttributeError`` from the call falls back to the Lightning path. No core module implements
    this today — it is the extension point for fully custom runners.
    """

    def run_teia_stage(self, **kwargs: Any) -> Any: ...


__all__ = [
    "BatchMetaSource",
    "TaskDataModule",
    "TeiaStageModule",
    "ArtifactPersistable",
    "InferSourceBindable",
    "TaskMetadataReady",
    "NativeStageRunner",
]
