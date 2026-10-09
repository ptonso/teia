from __future__ import annotations

import logging
from collections import namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lightning import LightningModule

from teia.base.envelope import check_envelope_collision
from teia.base.envelope import component_config as _object_cfg
from teia.core.module.codegen import (
    exec_and_bind,
    generate_activation_fn,
    generate_forward_fn,
    generate_losses_fn,
    sanitize_key,
    validate_pipeline,
)
from teia.core.capture.extract import CaptureMap
from teia.core.export.preprocess_slice import validate_export_kernels
from teia.core.module.dry_run import dry_run_pipeline

log = logging.getLogger(__name__)

#: Root-level ``netmodule.*`` keys that are module-wide config, not per-node aliases; the per-node
#: envelope vocabulary (``in``/``out``/``weight``/…) lives in the shared ``teia.base.envelope.ENVELOPE_KEYS``.
NETMODULE_ROOT_KEYS = {
    "_target_",
    "_recursive_",
    "optimizer_target",
    "lr_scheduler",
    "optimizers",
    "preset",
    "input",
    "capture",
}


@dataclass
class NodeBind:
    """Parameter-identity binding to another node (see teia:core/module/bind.md).

    ``mode='tied'`` — no module of its own; shares ``from`` node's live params (gradients
    flow back). ``mode='frozen'`` — a separate, non-trainable clone of ``from``'s module,
    slaved to it by ``sync`` (a ``SyncStrategy`` ``_target_`` spec). Absent ⇒ the node owns
    fresh trainable params (the default)."""

    from_node: str
    mode: str
    sync: dict[str, Any] | None = None


@dataclass
class PipelineNodeRecord:
    name: str
    in_key: list[str]
    out_key: list[str] | None
    out_shape: list[tuple[Any, ...] | str | None] | None
    is_loss: bool = False
    is_activation: bool = False
    weight: float = 1.0
    optimizer: str | int | None = None
    loss_routes: dict[str, str | int] = field(default_factory=dict)
    detach: list[str] = field(default_factory=list)
    aux_in: list[str] = field(default_factory=list)
    last_layer: str | None = None
    bind: NodeBind | None = None


def _normalize_in_key(raw_in: Any, *, name: str) -> list[str]:
    if raw_in is None:
        return []
    if isinstance(raw_in, str):
        return [raw_in]
    if isinstance(raw_in, Sequence):
        return [str(item) for item in raw_in]
    raise ValueError(
        f"Module alias '{name}' has invalid `in`: expected string or list, got {raw_in!r}"
    )


def _normalize_out_key(
    raw_out: Any,
    *,
    name: str,
) -> tuple[list[str] | None, list[tuple[Any, ...] | str | None] | None]:
    if raw_out is None:
        return None, None
    if not isinstance(raw_out, Mapping):
        raise ValueError(
            f"Module alias '{name}' has invalid `out`: expected a mapping of output keys to shapes or null."
        )

    out_key = list(raw_out.keys())
    out_shape: list[tuple[Any, ...] | str | None] = []
    for shape in raw_out.values():
        if shape is None:
            out_shape.append(None)
        elif isinstance(shape, str):
            out_shape.append(shape)
        elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes, bytearray)):
            values = tuple(shape)
            if len(values) == 1 and isinstance(values[0], str):
                out_shape.append(values[0])
            elif all(isinstance(dim, int) and not isinstance(dim, bool) for dim in values):
                out_shape.append(values)
            elif all(
                isinstance(dim, str)
                or (isinstance(dim, int) and not isinstance(dim, bool))
                for dim in values
            ):
                out_shape.append(values)
            else:
                raise ValueError(
                    f"Module alias '{name}' has invalid `out` shape {shape!r}: "
                    "expected null, an int list/tuple, or datamodule meta-token strings."
                )
        else:
            raise ValueError(
                f"Module alias '{name}' has invalid `out` shape {shape!r}: "
                "expected null, an int list/tuple, or a datamodule meta-token string."
            )
    return out_key, out_shape


def _shape_from_batch_meta(dm: Any, token: str) -> tuple[int, ...] | None:
    meta = dm.batch_meta()
    if token not in meta:
        return None
    shape = meta[token]
    if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"Datamodule batch_meta token '{token}' resolved to invalid shape {shape!r}.")
    return tuple(int(dim) for dim in shape)


def _resolve_out_shape_token(token: str, dm: Any, *, node_name: str) -> tuple[int, ...]:
    if dm is None:
        raise ValueError(f"Node '{node_name}': out_key token '{token}' requires a datamodule.")

    shape = _shape_from_batch_meta(dm, token)
    if shape is not None:
        return shape

    # Generic dim-ref fallback (C1): any datamodule attribute matching the token name
    # that resolves to an int (-> 1-tuple) or an int-tuple is a valid out_shape token. This is
    # the single open-extension path (spec-sanctioned, overview.md:73): it covers the standard
    # class-count tokens (``num_classes`` / ``num_targets`` / ``num_labels``) and pack-specific
    # dim-refs alike (the interactive regime's ``action_dim`` / ``num_actions`` / ``obs_state_dim`` / ``obs_pixels_shape``)
    # without a core whitelist edit.
    if hasattr(dm, token):
        resolved = _coerce_dimref_value(getattr(dm, token))
        if resolved is not None:
            return resolved

    raise ValueError(f"Node '{node_name}': unresolvable out_key meta-token '{token}'.")


def _coerce_dimref_value(value: Any) -> tuple[int, ...] | None:
    """Coerce a generic datamodule dim-ref attribute into an out_shape tuple.

    An int (or 0-d int-like) becomes a 1-tuple; an int sequence becomes that tuple.
    Anything else (None, str, float, nested) is rejected so the caller can raise a
    precise unresolved-token error. Booleans are not ints here.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
        if values and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in values):
            return tuple(int(dim) for dim in values)
    return None


def _batch_leading_dim(batch: Any) -> int:
    """Leading (batch) dimension read from the first non-``None`` tensor field (C2).

    ``batch[0]`` is unsafe for batch types with optional leading fields (e.g. the interactive regime's composed
    batch whose ``obs_state`` may be ``None`` for pixels-only presets). Scan in field order
    and use the first tensor that exposes a leading dim.
    """
    import torch

    fields = batch if isinstance(batch, (tuple, list)) else [batch]
    for field_value in fields:
        if torch.is_tensor(field_value) and field_value.ndim > 0:
            return int(field_value.shape[0])
    raise ValueError("Cannot determine batch size: no non-None tensor field found in batch.")


def _resolve_out_shape(shape: tuple[Any, ...] | str | None, dm: Any, *, node_name: str) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if isinstance(shape, str):
        return _resolve_out_shape_token(shape, dm, node_name=node_name)

    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, str):
            resolved.extend(_resolve_out_shape_token(dim, dm, node_name=node_name))
        else:
            resolved.append(int(dim))
    return tuple(resolved)


def _is_alias_entry(name: str, value: Any) -> bool:
    if name in NETMODULE_ROOT_KEYS or name.startswith("__"):
        return False
    if not isinstance(value, Mapping):
        return False
    # A node defines its own module (`_target_`) OR binds to another node's module
    # (`bind`, which clones/aliases the source — no `_target_` of its own).
    return "_target_" in value or isinstance(value.get("bind"), Mapping)


def _discover_alias_entries(kwargs: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    alias_entries: list[tuple[str, dict[str, Any]]] = []
    for key, value in kwargs.items():
        if key in {"pipeline", "losses"}:
            raise ValueError(
                "Legacy netmodule.pipeline/netmodule.losses configs are no longer supported. "
                "Import implementation YAMLs into top-level netmodule aliases and declare Teia graph "
                "metadata as flat `in`/`out`/… envelope keys on the alias."
            )

        if _is_alias_entry(key, value):
            alias_entries.append((key, dict(value)))
            continue

        if key in NETMODULE_ROOT_KEYS or key.startswith("__"):
            continue

        if isinstance(value, Mapping):
            raise ValueError(
                f"Module entry '{key}' must be a composed alias config with `_target_` and an `in`/`out` "
                "envelope. Import node parameters via the preset defaults list, then declare Teia-managed "
                f"metadata as flat keys on `netmodule.{key}`."
            )

        # Non-Mapping, non-reserved, non-alias entries are treated as optimizer kwargs — skip silently.

    return alias_entries


def _normalize_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    raise ValueError(f"Expected a string or list of strings, got {raw!r}.")


def _parse_loss_routes(raw_out: Mapping[str, Any], *, name: str) -> tuple[list[str], dict[str, str | int]]:
    """For a loss node the out map value is the optimizer route (name or index), not a shape.

    ``null`` means the term follows the node's ``optimizer`` key (then index 0); see
    teia:core/module/optimization.md §3.4.
    """
    out_key = list(raw_out.keys())
    routes: dict[str, str | int] = {}
    for key, value in raw_out.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(
                f"Module alias '{name}' loss out '{key}' route must be an optimizer name, "
                f"an integer index, or null; got {value!r}."
            )
        routes[key] = value
    return out_key, routes


def _resolve_alias_class(entry: Mapping[str, Any]) -> type | None:
    target = entry.get("_target_")
    if not target:
        return None
    from teia.core.utils import import_string

    try:
        cls = import_string(str(target))
    except Exception:
        return None
    return cls if isinstance(cls, type) else None


def _parse_node_record(name: str, entry: Mapping[str, Any]) -> PipelineNodeRecord:
    check_envelope_collision(_resolve_alias_class(entry), entry, name=name)

    in_key = _normalize_in_key(entry.get("in"), name=name)
    raw_out = entry.get("out")
    out_keys = list(raw_out.keys()) if isinstance(raw_out, Mapping) else []
    is_loss = any(k.startswith("loss.") for k in out_keys)
    is_activation = any(k.startswith("act.") for k in out_keys)

    loss_routes: dict[str, str | int] = {}
    if is_loss:
        out_key, loss_routes = _parse_loss_routes(raw_out, name=name)
        out_shape: list[tuple[Any, ...] | str | None] | None = None
    else:
        out_key, out_shape = _normalize_out_key(raw_out, name=name)

    return PipelineNodeRecord(
        name=str(name).replace("/", "_").replace("-", "_"),
        in_key=in_key,
        out_key=out_key,
        out_shape=out_shape,
        is_loss=is_loss,
        is_activation=is_activation,
        weight=float(entry.get("weight", 1.0)),
        optimizer=entry.get("optimizer"),
        loss_routes=loss_routes,
        detach=_normalize_str_list(entry.get("detach")),
        aux_in=_normalize_str_list(entry.get("aux_in")),
        last_layer=entry.get("last_layer"),
        bind=_parse_bind(entry.get("bind"), name=name),
    )


def _parse_bind(raw: Any, *, name: str) -> NodeBind | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"Module alias '{name}' bind must be a mapping, got {raw!r}.")
    from_node = raw.get("from")
    mode = raw.get("mode")
    if not from_node:
        raise ValueError(f"Module alias '{name}' bind requires `from` (a source node name).")
    if mode not in ("tied", "frozen"):
        raise ValueError(f"Module alias '{name}' bind.mode must be 'tied' or 'frozen', got {mode!r}.")
    sync = raw.get("sync")
    if mode == "frozen" and not (isinstance(sync, Mapping) and "_target_" in sync):
        raise ValueError(
            f"Module alias '{name}' bind.mode='frozen' requires a `sync` SyncStrategy "
            f"_target_ spec (e.g. teia.core.module.sync.HardSync/EmaSync)."
        )
    return NodeBind(
        from_node=str(from_node).replace("/", "_").replace("-", "_"),
        mode=str(mode),
        sync=dict(sync) if isinstance(sync, Mapping) else None,
    )


def _toposort_pipeline(
    pairs: list[tuple[PipelineNodeRecord, dict[str, Any]]],
) -> list[tuple[PipelineNodeRecord, dict[str, Any]]]:
    if not pairs:
        return []

    pair_by_name = {rec.name: (rec, entry) for rec, entry in pairs}
    order_index = {rec.name: idx for idx, (rec, _) in enumerate(pairs)}

    producer_by_key: dict[str, str] = {}
    for rec, _ in pairs:
        if not rec.out_key:
            continue
        for key in rec.out_key:
            if key in producer_by_key:
                raise ValueError(f"Duplicate out_key '{key}' in module graph.")
            producer_by_key[key] = rec.name

    deps: dict[str, set[str]] = {rec.name: set() for rec, _ in pairs}
    consumers: dict[str, set[str]] = {rec.name: set() for rec, _ in pairs}
    for rec, _ in pairs:
        for key in rec.in_key:
            if key.startswith("batch."):
                continue
            producer = producer_by_key.get(key)
            if producer is None:
                raise ValueError(
                    f"Node '{rec.name}': in_key '{key}' is not defined by any node out_key and is not a batch.* key."
                )
            deps[rec.name].add(producer)
            consumers[producer].add(rec.name)
        # A bind node must be built/run after its source node (so the source module exists to
        # alias or clone) even when it has no dataflow edge to it.
        if rec.bind is not None:
            source = rec.bind.from_node
            if source not in deps:
                raise ValueError(f"Node '{rec.name}': bind.from '{source}' is not a forward node in the graph.")
            deps[rec.name].add(source)
            consumers[source].add(rec.name)

    ready = sorted((name for name, incoming in deps.items() if not incoming), key=order_index.get)
    ordered_names: list[str] = []

    while ready:
        current = ready.pop(0)
        ordered_names.append(current)
        for consumer in sorted(consumers[current], key=order_index.get):
            deps[consumer].discard(current)
            if not deps[consumer]:
                ready.append(consumer)
        ready.sort(key=order_index.get)

    if len(ordered_names) != len(pairs):
        cyclic = sorted((name for name, incoming in deps.items() if incoming), key=order_index.get)
        raise ValueError(f"Cycle detected in module graph involving: {', '.join(cyclic)}")

    return [pair_by_name[name] for name in ordered_names]


def _instantiate_if_spec(value: Any) -> Any:
    """Instantiate an optimizer kwarg that is itself a ``{_target_: ...}`` object spec.

    Lets an optimizer take an injected collaborator by config (e.g. the trust-region optimizer's
    ``objective`` provider). Scalar/sequence kwargs pass through unchanged."""
    from teia.core.instantiate import instantiate

    if isinstance(value, Mapping) and "_target_" in value:
        return instantiate(value)
    return value


def _instantiate_scheduler_value(value: Any, optimizer: Any) -> Any:
    from teia.core.instantiate import instantiate

    if isinstance(value, Mapping):
        if "_target_" in value:
            cfg = {
                key: _instantiate_scheduler_value(item, optimizer)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
            cfg.update({key: item for key, item in value.items() if str(key).startswith("_")})
            target = str(cfg.get("_target_", ""))
            if target.startswith("torch.optim.lr_scheduler.") and "optimizer" not in cfg:
                cfg["optimizer"] = optimizer
            return instantiate(cfg)
        return {key: _instantiate_scheduler_value(item, optimizer) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_instantiate_scheduler_value(item, optimizer) for item in value]

    return value


def _is_teia_node_target(cfg: dict[str, Any]) -> bool:
    from teia.base.net import TeiaNode
    from teia.core.utils import import_string

    target = cfg.get("_target_", "")
    if not target:
        return False
    try:
        cls = import_string(str(target))
        return isinstance(cls, type) and issubclass(cls, TeiaNode)
    except Exception:
        return False


def _meta_forward(node: Any, rec: PipelineNodeRecord, workspace: dict[str, Any]) -> None:
    import torch

    if not rec.in_key or not rec.out_key:
        return
    missing = [key for key in rec.in_key if key not in workspace]
    if missing:
        raise RuntimeError(
            f"Meta dry-run failed for node '{rec.name}': missing workspace keys {missing}. "
            f"Available: {list(workspace.keys())}"
        )

    args = [workspace[key] for key in rec.in_key]
    node.eval()
    with torch.no_grad():
        try:
            result = node(*args)
        except Exception as exc:
            raise RuntimeError(f"Meta dry-run failed for node '{rec.name}': {exc}") from exc
    node.train()

    results = (result,) if not isinstance(result, (tuple, list)) else tuple(result)
    for key, tensor in zip(rec.out_key, results):
        workspace[key] = tensor


def _make_pred_namedtuple(pred_fields: list[str]):
    field_names = [field[len("pred."):] if field.startswith("pred.") else field for field in pred_fields]
    return namedtuple("Pred", field_names)


def _validate_batch_workspace(records: list[PipelineNodeRecord], workspace: dict[str, Any]) -> None:
    for rec in records:
        missing = [key for key in rec.in_key if key.startswith("batch.") and key not in workspace]
        if missing:
            raise ValueError(
                f"Node '{rec.name}': batch in_key {missing} missing from datamodule batch_meta(). "
                f"Available batch keys: {sorted(key for key in workspace if key.startswith('batch.'))}"
            )


class TeiaNetModule(LightningModule):
    def __init__(
        self,
        optimizer_target: str = "torch.optim.SGD",
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Always manual (single- and multi-optimizer alike); grad clip / accumulation applied here,
        # not by the trainer, which Lightning forbids in this mode.
        self.automatic_optimization = False

        # Runtime state seeded up-front so every reader is a plain attribute access (no
        # ``getattr``/``hasattr`` self-probing). ``_build_nodes`` / the trainer hand-off /
        # ``configure_from_datamodule`` overwrite these once the real values are known.
        self._param_syncs: list[tuple[str, str, Any]] = []
        self._manual_accumulate_grad_batches: int | None = None
        self._manual_grad_clip_val: float | None = None
        self._manual_grad_clip_algorithm: str = "norm"
        self._epoch_optimizer_steps: set[str] = set()
        self._scheduler_runtime: list[dict[str, Any]] = []
        self._predict_ctx_builder: Any = None
        #: Set by the engine: the task contract (or ``None``), the evalmodule's ``batch.*`` reads for a
        #: task-less run, and the postprocess ctx capture uses (``None`` = no capture this run).
        self.contract: Any = None
        self.eval_batch_keys: set[str] = set()
        self.capture_ctx: dict[str, Any] | None = None
        self.infer_batch_keys: list[str] = []

        self._optimizer_target = optimizer_target
        raw_lr_scheduler = kwargs.get("lr_scheduler")
        if raw_lr_scheduler is None:
            self._lr_scheduler_config: dict[str, Any] | None = None
        elif isinstance(raw_lr_scheduler, Mapping):
            self._lr_scheduler_config = dict(raw_lr_scheduler)
        else:
            raise TypeError("netmodule.lr_scheduler must be a mapping matching Lightning lr_scheduler_config.")

        self._capture_spec: dict[str, str] = {str(k): str(v) for k, v in dict(kwargs.get("capture") or {}).items()}

        alias_entries = _discover_alias_entries(kwargs)
        alias_names = {name for name, _ in alias_entries}
        self._optimizer_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in NETMODULE_ROOT_KEYS
            and k not in alias_names
            and not isinstance(v, Mapping)
            and not k.startswith("__")
        }
        forward_pairs: list[tuple[PipelineNodeRecord, dict[str, Any]]] = []
        activation_pairs: list[tuple[PipelineNodeRecord, dict[str, Any]]] = []
        loss_pairs: list[tuple[PipelineNodeRecord, dict[str, Any]]] = []

        for name, entry in alias_entries:
            record = _parse_node_record(name, entry)
            if record.is_loss:
                loss_pairs.append((record, entry))
            elif record.is_activation:
                activation_pairs.append((record, entry))
            else:
                forward_pairs.append((record, entry))

        ordered_forward_pairs = _toposort_pipeline(forward_pairs)
        self._pipeline_records = [rec for rec, _ in ordered_forward_pairs]
        self._pipeline_entries = [entry for _, entry in ordered_forward_pairs]
        self._activation_records = [rec for rec, _ in activation_pairs]
        self._activation_entries = [entry for _, entry in activation_pairs]
        self._loss_records = [rec for rec, _ in loss_pairs]
        self._loss_entries = [entry for _, entry in loss_pairs]

        self._build_optimizer_routing(kwargs.get("optimizers"))
        self._finish_build()

    def _finish_build(self) -> None:
        pred_fields: list[str] = []
        for rec in self._pipeline_records:
            if rec.out_key is None:
                continue
            pred_fields.extend(key for key in rec.out_key if key.startswith("pred."))

        self._pred_fields = pred_fields
        self._pred_type = _make_pred_namedtuple(pred_fields)

        validate_pipeline(self._pipeline_records, self._loss_records, self._activation_records)
        CaptureMap(self._capture_spec, self.activation_outputs(), None)  # fail fast on bad sources
        exec_and_bind(
            generate_forward_fn(self._pipeline_records, pred_fields),
            self,
            "_generated_forward",
        )
        exec_and_bind(
            generate_losses_fn(self._loss_records),
            self,
            "_generated_losses",
        )
        exec_and_bind(
            generate_activation_fn(self._activation_records),
            self,
            "_generated_activation",
        )

    def _build_optimizer_routing(self, raw_optimizers: Any) -> None:
        """Parse ``optimizers`` (or desugar the flat form) into specs, the node→optimizer ownership
        map, and the per-loss-term route + weight maps."""
        self._optimizer_specs = self._normalize_optimizer_specs(raw_optimizers)
        self._optimizer_names = [spec["name"] for spec in self._optimizer_specs]
        if len(set(self._optimizer_names)) != len(self._optimizer_names):
            raise ValueError(f"Duplicate optimizer names in netmodule.optimizers: {self._optimizer_names}.")
        default = self._optimizer_names[0]

        self._node_optimizer: dict[str, str] = {}
        for rec in [*self._pipeline_records, *self._loss_records, *self._activation_records]:
            self._node_optimizer[rec.name] = self._resolve_optimizer_name(rec.optimizer, default=default)

        self._loss_term_route: dict[str, str] = {}
        self._loss_term_weight: dict[str, float] = {}
        for rec in self._loss_records:
            node_default = self._node_optimizer[rec.name]
            for key in rec.out_key or []:
                route = rec.loss_routes.get(key)
                self._loss_term_route[key] = (
                    node_default if route is None else self._resolve_optimizer_name(route, default=node_default)
                )
                self._loss_term_weight[key] = rec.weight

    def _normalize_optimizer_specs(self, raw_optimizers: Any) -> list[dict[str, Any]]:
        if raw_optimizers is None:
            return [
                {
                    "name": "default",
                    "_target_": self._optimizer_target,
                    "kwargs": dict(self._optimizer_kwargs),
                    "start_step": 0,
                    "every": 1,
                    "grad_clip": None,
                    "scheduler": self._lr_scheduler_config,
                }
            ]
        if not isinstance(raw_optimizers, Sequence) or isinstance(raw_optimizers, (str, bytes)):
            raise TypeError("netmodule.optimizers must be a list of named optimizer specs.")

        specs: list[dict[str, Any]] = []
        for idx, raw in enumerate(raw_optimizers):
            if not isinstance(raw, Mapping):
                raise TypeError(f"netmodule.optimizers[{idx}] must be a mapping.")
            entry = dict(raw)
            target = entry.pop("_target_", None)
            if not target:
                raise ValueError(f"netmodule.optimizers[{idx}] is missing `_target_`.")
            name = entry.pop("name", None)
            if name is None:
                if len(raw_optimizers) > 1:
                    raise ValueError("netmodule.optimizers entries require `name` when more than one is declared.")
                name = "default"
            scheduler = entry.pop("scheduler", None)
            if scheduler is not None and not isinstance(scheduler, Mapping):
                raise TypeError(f"netmodule.optimizers[{idx}].scheduler must be a mapping.")
            grad_clip = entry.pop("grad_clip", None)
            start_step = int(entry.pop("start_step", 0) or 0)
            every = int(entry.pop("every", 1) or 1)
            specs.append(
                {
                    "name": str(name),
                    "_target_": str(target),
                    "kwargs": entry,  # remaining flat keys: lr, weight_decay, betas, decouple_weight_decay
                    "start_step": start_step,
                    "every": every,
                    "grad_clip": dict(grad_clip) if grad_clip else None,
                    "scheduler": dict(scheduler) if scheduler else None,
                }
            )
        return specs

    def _resolve_optimizer_name(self, value: Any, *, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError(f"optimizer must be a name or index, got bool {value!r}.")
        if isinstance(value, int):
            if not 0 <= value < len(self._optimizer_names):
                raise ValueError(f"optimizer index {value} out of range for {self._optimizer_names}.")
            return self._optimizer_names[value]
        if isinstance(value, str):
            if value not in self._optimizer_names:
                raise ValueError(f"optimizer '{value}' is not a declared optimizer {self._optimizer_names}.")
            return value
        raise ValueError(f"optimizer must be a name or index, got {value!r}.")

    def _resolve_last_layer(self, alias: str):
        """A forward alias's reference param for the adaptive weight: the last trainable weight
        tensor (ndim >= 2, e.g. the decoder's final conv weight), else its last trainable param."""
        node = getattr(self, sanitize_key(alias), None)
        if node is None:
            raise ValueError(f"last_layer alias '{alias}' has no built node.")
        params = [p for p in node.parameters() if p.requires_grad]
        if not params:
            raise ValueError(f"last_layer alias '{alias}' exposes no trainable parameters.")
        for param in reversed(params):
            if param.ndim >= 2:
                return param
        return params[-1]

    def _build_nodes(self, workspace: dict[str, Any], dm: Any = None) -> None:
        from hydra.utils import instantiate as hydra_instantiate

        self._param_syncs: list[tuple[str, str, Any]] = []
        producer_by_key: dict[str, Any] = {}

        def _configure_producers(node: Any, rec: PipelineNodeRecord) -> None:
            if not hasattr(node, "configure_from_producers"):
                return
            producers = {key: producer_by_key[key] for key in rec.in_key if key in producer_by_key}
            node.configure_from_producers(producers)

        for rec, entry in zip(self._pipeline_records, self._pipeline_entries):
            attr = sanitize_key(rec.name)
            if rec.bind is not None:
                node = self._build_bind_node(rec, workspace)
                for key in rec.out_key or []:
                    producer_by_key[key] = node
                continue
            cfg = _object_cfg(entry)

            missing = [key for key in rec.in_key if key not in workspace]
            if missing:
                raise RuntimeError(
                    f"Node '{rec.name}': missing in_key {missing}. "
                    f"Available: {list(workspace.keys())}"
                )
            in_shapes = [tuple(workspace[key].shape[1:]) for key in rec.in_key]
            final_in = in_shapes[0] if len(in_shapes) == 1 else in_shapes

            final_out: Any = None
            if rec.out_shape is not None:
                resolved_out = [
                    _resolve_out_shape(shape, dm, node_name=rec.name)
                    for shape in rec.out_shape
                ]
                final_out = resolved_out[0] if len(resolved_out) == 1 else resolved_out

            if _is_teia_node_target(cfg):
                node = hydra_instantiate(
                    cfg,
                    _convert_="partial",
                    in_shape=final_in,
                    out_shape=final_out,
                )
            else:
                node = hydra_instantiate(cfg)

            setattr(self, attr, node)
            if dm is not None and hasattr(node, "configure_from_datamodule"):
                node.configure_from_datamodule(dm)
            _configure_producers(node, rec)
            _meta_forward(node, rec, workspace)
            for key in rec.out_key or []:
                producer_by_key[key] = node

        for rec, entry in zip(self._loss_records, self._loss_entries):
            attr = sanitize_key(rec.name)
            node = hydra_instantiate(_object_cfg(entry))
            setattr(self, attr, node)
            _configure_producers(node, rec)
            for key in rec.out_key or []:
                producer_by_key[key] = node

        for rec, entry in zip(self._activation_records, self._activation_entries):
            attr = sanitize_key(rec.name)
            node = hydra_instantiate(_object_cfg(entry))
            setattr(self, attr, node)
            if dm is not None and hasattr(node, "configure_from_datamodule"):
                node.configure_from_datamodule(dm)
            _configure_producers(node, rec)
            for key in rec.out_key or []:
                producer_by_key[key] = node


    def _build_bind_node(self, rec: PipelineNodeRecord, workspace: dict[str, Any]) -> Any:
        """Build a ``bind`` node: ``tied`` shares the source's live module (no params of its
        own — the generated forward calls the source directly); ``frozen`` registers a separate,
        non-trainable deep-copy of the source (target = source at init), slaved each step by its
        ``sync`` strategy. See teia:core/module/bind.md. Returns the effective node (``source``
        for ``tied``, the clone for ``frozen``) so callers can register it as a producer."""
        import copy

        from hydra.utils import instantiate as hydra_instantiate

        source = getattr(self, sanitize_key(rec.bind.from_node), None)
        if source is None:
            raise RuntimeError(
                f"Node '{rec.name}': bind source '{rec.bind.from_node}' has no built module "
                f"(it must be a forward node built before this one)."
            )
        if rec.bind.mode == "tied":
            _meta_forward(source, rec, workspace)  # propagate this call site's output shape
            return source
        clone = copy.deepcopy(source)
        for param in clone.parameters():
            param.requires_grad_(False)
        setattr(self, sanitize_key(rec.name), clone)
        strategy = hydra_instantiate(rec.bind.sync)
        self._param_syncs.append((sanitize_key(rec.name), sanitize_key(rec.bind.from_node), strategy))
        return clone
        _meta_forward(clone, rec, workspace)

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        """Drive intrinsic parameter syncs for ``frozen`` bind nodes (auto-paired from the graph).

        Replaces the external target-sync callback: the module owns its parameter graph, so it
        syncs its own frozen clones — you can't forget to wire it. Runs after the optimizer step,
        so frozen targets track the just-updated source weights."""
        if not self._param_syncs:
            return
        global_step = int(getattr(self, "global_step", 0))
        for target_attr, source_attr, strategy in self._param_syncs:
            strategy.sync(getattr(self, target_attr), getattr(self, source_attr), global_step=global_step)

    def forward(self, batch: Any):
        return self._generated_forward(batch)

    def _node(self, rec: PipelineNodeRecord) -> Any:
        """Fail-fast accessor for a record's built node — post-build it MUST exist."""
        node = getattr(self, sanitize_key(rec.name), None)
        if node is None:
            raise RuntimeError(f"Node '{rec.name}' has no built module; build the pipeline first.")
        return node

    def capture_map_keys(self) -> list[str]:
        """The ``route.atom`` keys of the netmodule's ``capture:`` map."""
        return list(self._capture_spec)

    def activation_outputs(self) -> dict[str, str]:
        """Each ``act.*`` out-key → the activation alias producing it."""
        return {key: rec.name for rec in self._activation_records for key in (rec.out_key or [])}

    @property
    def capture_map(self) -> CaptureMap:
        """Boundary B: the ``capture:`` map typed by the task contract (teia:core/module/capture_map.md)."""
        return CaptureMap(self._capture_spec, self.activation_outputs(), self.contract, self.eval_batch_keys)

    def step_atoms(self, batch: Any, pred: Any, ctx: Mapping[str, Any], *, batch_keys: list[str] | None = None) -> dict[str, Any]:
        """``capture.*`` atoms (+ target ``batch.*`` fields) for one step, through the capture map."""
        capture = self.capture_map
        activated = self.activate(pred)
        post = {
            rec.name: self._node(rec).postprocess(activated[rec.name], {**ctx, **self._node(rec).params()})
            for rec in self._activation_records
            if rec.name in capture.post_aliases()
        }
        return capture.atoms(batch, activated, post, batch_keys)

    def activate(self, pred: Any) -> dict[str, dict[str, Any]]:
        """Run each activation node's ``activation`` over the ``pred.*`` logits.

        Returns ``{act_alias: {activated_key: Tensor}}`` — the pure-tensor outputs
        that are also traced into the ONNX graph (see teia:core/module/activation_route.md).
        """
        return self._generated_activation(pred)

    def postprocess(self, activated: dict[str, dict[str, Any]], ctx: Mapping[str, Any]) -> dict[str, Any]:
        """Run each activation node's host-side ``postprocess`` over the activated outputs.

        Each node's own ``params()`` literals (e.g. ``QuantileActivation.quantiles``) are merged
        into its ctx, since ``decode`` is a bare staticmethod that cannot read ``self``. Returns a
        dict keyed by activation-node name (one route per key).
        """
        out: dict[str, Any] = {}
        for rec in self._activation_records:
            node = self._node(rec)
            node_ctx = {**ctx, **node.params()}
            out[rec.name] = node.postprocess(activated.get(rec.name, {}), node_ctx)
        return out

    def _step(self, batch: Any, stage: str):
        """Validation / test step — no optimization; logs the summed total.

        When the datamodule flags capture-only evaluation (``dm.eval_capture_only``)
        the step runs ``forward`` + report-capture but skips loss generation and the
        route sum (C3). RL eval uses this: eval batches are full forward-valid values
        but carry no loss-only fields (``advantage``/``log_prob``).
        """
        self._sync_global_step_to_nodes()
        pred = self(batch)
        batch_size = _batch_leading_dim(batch)

        output: dict[str, Any] = {}
        if not self._is_capture_only_eval():
            loss_log = self._generated_losses(batch, pred)
            for key, value in loss_log.items():
                if value is not None:
                    self.log(
                        f"{stage}/{key}",
                        value,
                        prog_bar=(key == "loss"),
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size,
                    )
            output["loss"] = loss_log["loss"]

        if self.capture_ctx is not None:
            output["atoms"] = self.step_atoms(batch, pred, self.capture_ctx)
            output["n_samples"] = batch_size
        return output

    def _in_fit_validation(self) -> bool:
        """True only inside ``trainer.fit``'s validation loop (not standalone validate/test)."""
        trainer = getattr(self, "_trainer", None)
        fn = getattr(getattr(trainer, "state", None), "fn", None)
        return str(getattr(fn, "value", fn)).lower() == "fit"

    def _is_capture_only_eval(self) -> bool:
        """Read the datamodule capture-only eval flag (set by the RL eval dataloader).

        Reads ``self._trainer`` (the raw backing attribute) rather than the public
        ``trainer`` property, which *raises* when the module is not attached to a
        Trainer instead of returning ``None``.
        """
        trainer = getattr(self, "_trainer", None)
        dm = getattr(trainer, "datamodule", None)
        return bool(getattr(dm, "eval_capture_only", False))

    def _sync_global_step_to_nodes(self) -> None:
        """Hand the current global step to loss nodes that gate on it (e.g. the adversarial warmup)."""
        gs = int(self.global_step)
        for rec in self._loss_records:
            node = self._node(rec)
            if hasattr(node, "global_step"):  # genuine optional: only step-gated losses expose it
                node.global_step = gs

    def _route_sums(self, loss_log: Mapping[str, Any]) -> dict[str, Any]:
        """Group per-term ``loss.*`` tensors into one weighted backward target per optimizer route."""
        sums: dict[str, Any] = {}
        for key, value in loss_log.items():
            if key == "loss" or not key.startswith("loss.") or value is None:
                continue
            opt_name = self._loss_term_route[key]
            term = self._loss_term_weight[key] * value
            sums[opt_name] = term if opt_name not in sums else sums[opt_name] + term
        return sums

    def training_step(self, batch: Any, batch_idx: int):
        """Manual step: one shared forward, then per active route in declared order
        toggle → manual_backward (retain_graph for all but the last) → clip → step → zero_grad →
        untoggle; honors gradient accumulation and steps step-interval schedulers."""
        import torch

        optimizers = self.optimizers()
        optimizers = list(optimizers) if isinstance(optimizers, (list, tuple)) else [optimizers]

        self._sync_global_step_to_nodes()
        pred = self(batch)
        loss_log = self._generated_losses(batch, pred)
        route_sums = self._route_sums(loss_log)

        accumulate = max(int(self._accumulate_grad_batches()), 1)
        do_step = (batch_idx + 1) % accumulate == 0 or self._is_last_train_batch(batch_idx)
        gs = int(self.global_step)
        stepped_optimizers: set[str] = set()

        # A closure-driven optimizer (trust-region / LBFGS) owns its own backward + step and so
        # is active on cadence regardless of whether it contributes a `loss.*` route term.
        active = [
            (idx, spec)
            for idx, spec in enumerate(self._optimizer_specs)
            if (spec["name"] in route_sums or getattr(optimizers[idx], "requires_closure", False))
            and gs >= spec["start_step"]
            and gs % spec["every"] == 0
        ]
        for pos, (idx, spec) in enumerate(active):
            opt = optimizers[idx]
            is_last = pos == len(active) - 1
            if getattr(opt, "requires_closure", False):
                # The loop binds (module, batch); the optimizer re-evaluates the injected
                # objective at candidate params during its line search. See teia:core/module/optimization.md.
                self.toggle_optimizer(opt)
                if do_step:
                    opt.step(lambda opt=opt: opt.objective.evaluate(self, batch))
                    opt.zero_grad()
                    stepped_optimizers.add(spec["name"])
                self.untoggle_optimizer(opt)
                continue
            route_loss = route_sums[spec["name"]] / accumulate
            self.toggle_optimizer(opt)
            self.manual_backward(route_loss, retain_graph=not is_last)
            if do_step:
                self._clip_route_gradients(opt, spec)
                opt.step()
                opt.zero_grad()
                stepped_optimizers.add(spec["name"])
            self.untoggle_optimizer(opt)

        if stepped_optimizers:
            self._epoch_optimizer_steps.update(stepped_optimizers)
            self._step_schedulers(interval="step", optimizer_names=stepped_optimizers)

        batch_size = _batch_leading_dim(batch)
        for key, value in loss_log.items():
            if value is not None and torch.is_tensor(value):
                self.log(
                    f"train/{key}",
                    value,
                    prog_bar=(key == "loss"),
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )

    def _accumulate_grad_batches(self) -> int:
        # The runtime stashes the configured value here (manual mode ignores trainer accumulation).
        if self._manual_accumulate_grad_batches is not None:
            return int(self._manual_accumulate_grad_batches)
        return int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)

    def _is_last_train_batch(self, batch_idx: int) -> bool:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return False
        try:
            if bool(getattr(trainer, "is_last_batch", False)):
                return True
        except Exception:
            pass
        try:
            num_batches = getattr(trainer, "num_training_batches", None)
            if num_batches is None:
                return False
            if not isinstance(num_batches, int):
                import math

                if not math.isfinite(float(num_batches)):
                    return False
                num_batches = int(num_batches)
            return num_batches > 0 and (batch_idx + 1) >= num_batches
        except Exception:
            return False

    def _clip_route_gradients(self, optimizer: Any, spec: Mapping[str, Any]) -> None:
        grad_clip = spec.get("grad_clip")
        if grad_clip:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=grad_clip.get("val"),
                gradient_clip_algorithm=grad_clip.get("algorithm", "norm"),
            )
            return
        if self._manual_grad_clip_val:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self._manual_grad_clip_val,
                gradient_clip_algorithm=self._manual_grad_clip_algorithm,
            )

    def _step_schedulers(self, *, interval: str, optimizer_names: set[str] | None = None) -> None:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        for entry in self._scheduler_runtime:
            if entry["interval"] != interval:
                continue
            if optimizer_names is not None and entry["optimizer_name"] not in optimizer_names:
                continue
            counter = self.global_step if interval == "step" else self.current_epoch
            if (int(counter) + 1) % entry["frequency"] != 0:
                continue
            scheduler = entry["scheduler"]
            if isinstance(scheduler, ReduceLROnPlateau):
                monitor = entry.get("monitor")
                metric = self.trainer.callback_metrics.get(monitor) if monitor else None
                if metric is not None:
                    scheduler.step(metric)
            else:
                scheduler.step()

    def on_train_epoch_end(self) -> None:
        if self._epoch_optimizer_steps:
            self._step_schedulers(interval="epoch", optimizer_names=set(self._epoch_optimizer_steps))
        self._epoch_optimizer_steps.clear()

    def validation_step(self, batch: Any, batch_idx: int):
        return self._step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Infer route under ``trainer.predict``: ``capture.*`` atoms through the capture map, with a
        host-injected ``_predict_ctx_builder`` supplying the per-batch ctx (``infer.ctx`` + restore)."""
        builder = self._predict_ctx_builder
        ctx = builder(batch) if callable(builder) else {}
        return {
            "atoms": self.step_atoms(batch, self(batch), ctx, batch_keys=self.infer_batch_keys),
            "n_samples": _batch_leading_dim(batch),
        }

    def configure_optimizers(self):
        """One optimizer per spec over its assigned nodes' params (index 0 absorbs unclaimed params);
        norms+bias no-decay split applied per optimizer. Single optimizer returns the plain /
        dict form, multiple return the Lightning list form. A netmodule with no trainable parameter
        (a frozen model) gets no optimizer, and is only valid when nothing trains (``epochs=0``)."""
        params_by_optimizer = self._collect_params_by_optimizer()
        if not any(params_by_optimizer.values()):
            if self.trainer.max_epochs != 0:
                raise ValueError("the netmodule has no trainable parameter; a frozen model runs with epochs=0.")
            return None
        self._scheduler_runtime: list[dict[str, Any]] = []
        self._epoch_optimizer_steps: set[str] = set()

        optimizers = []
        scheduler_configs = []
        for spec in self._optimizer_specs:
            optimizer = self._build_one_optimizer(spec, params_by_optimizer[spec["name"]])
            optimizers.append(optimizer)
            if spec.get("scheduler"):
                scheduler_configs.append(
                    self._build_scheduler_config(spec["scheduler"], optimizer, optimizer_name=spec["name"])
                )

        if len(optimizers) == 1:
            if not scheduler_configs:
                return optimizers[0]
            return {"optimizer": optimizers[0], "lr_scheduler": scheduler_configs[0]}

        if scheduler_configs:
            return optimizers, scheduler_configs
        return optimizers

    def _collect_params_by_optimizer(self) -> dict[str, list]:
        by_name: dict[str, list] = {name: [] for name in self._optimizer_names}
        default = self._optimizer_names[0]
        claimed: set[int] = set()
        for rec in [*self._pipeline_records, *self._loss_records, *self._activation_records]:
            node = getattr(self, sanitize_key(rec.name), None)
            if node is None:
                continue
            owner = self._node_optimizer.get(rec.name, default)
            for param in node.parameters():
                if id(param) in claimed:
                    continue
                claimed.add(id(param))
                by_name[owner].append(param)
        for param in self.parameters():  # default absorbs any parameter not owned by a node
            if id(param) in claimed:
                continue
            claimed.add(id(param))
            by_name[default].append(param)
        return by_name

    def _build_one_optimizer(self, spec: Mapping[str, Any], params: list):
        from teia.core.utils import import_string

        opt_cls = import_string(spec["_target_"])
        opt_kwargs = {key: _instantiate_if_spec(value) for key, value in dict(spec["kwargs"]).items()}
        decouple = bool(opt_kwargs.pop("decouple_weight_decay", True))
        wd = float(opt_kwargs.get("weight_decay", 0.0) or 0.0)
        trainable = [p for p in params if p.requires_grad]
        if decouple and wd > 0.0:
            decay_params = [p for p in trainable if p.ndim >= 2]
            no_decay_params = [p for p in trainable if p.ndim < 2]
            param_groups = [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            opt_kwargs.pop("weight_decay", None)
            return opt_cls(param_groups, **opt_kwargs)
        return opt_cls(trainable, **opt_kwargs)

    def _build_scheduler_config(
        self,
        raw_scheduler: Mapping[str, Any],
        optimizer: Any,
        *,
        optimizer_name: str,
    ) -> dict[str, Any]:
        if "scheduler" not in raw_scheduler:
            raise ValueError("An optimizer scheduler config must include a 'scheduler' entry.")
        scheduler = _instantiate_scheduler_value(raw_scheduler["scheduler"], optimizer)
        config = {"scheduler": scheduler}
        for key in ("interval", "frequency", "monitor", "strict", "name"):
            if key in raw_scheduler:
                config[key] = raw_scheduler[key]
        # Manual optimization ignores interval/frequency; record them so the module steps manually.
        self._scheduler_runtime.append(
            {
                "scheduler": scheduler,
                "interval": str(raw_scheduler.get("interval", "epoch")),
                "frequency": max(int(raw_scheduler.get("frequency", 1) or 1), 1),
                "monitor": raw_scheduler.get("monitor"),
                "optimizer_name": optimizer_name,
            }
        )
        return config

    def configure_from_datamodule(self, dm: Any) -> None:
        import torch

        meta: dict[str, Any] = dm.batch_meta()
        workspace: dict[str, Any] = {
            f"batch.{key}": torch.zeros(1, *shape)
            for key, shape in meta.items()
            if isinstance(shape, (tuple, list)) and all(isinstance(dim, int) for dim in shape)
        }
        _validate_batch_workspace(
            [*self._pipeline_records, *self._loss_records, *self._activation_records],
            workspace,
        )

        self._build_nodes(workspace, dm=dm)
        self._finish_build()
        validate_export_kernels(dm, self)

        for rec in [*self._loss_records, *self._activation_records]:
            node = self._node(rec)
            if hasattr(node, "configure_from_datamodule"):
                node.configure_from_datamodule(dm)

    def on_validation_model_train(self) -> None:
        self.train()

    def on_train_start(self) -> None:
        self.train()

    @property
    def allow_eval_modules_at_train_start(self) -> bool:
        for rec in self._pipeline_records:
            node = getattr(self, sanitize_key(rec.name), None)
            if node is not None and getattr(node, "freeze", False):
                return True
        return False

    def dry_run(self, dm: Any) -> None:
        dry_run_pipeline(self, dm)
