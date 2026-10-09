from __future__ import annotations

import inspect
import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from teia.core.deps import MissingDependencyError
from teia.core.export.assemble import write_bundle_runtime
from teia.core.export.bundle import BundleContract, BundlePaths, InputSpec, write_bundle_readme, write_bundle_spec
from teia.core.export.preprocess_slice import PreprocessStep, resolve_chain_for, resolve_forward_batch_fields
from teia.core.module.codegen import sanitize_key
from teia.core.torch_compat import load_trusted_checkpoint
from teia.core.utils import ensure_dir, teia_log


@contextmanager
def suppress_export_noise(*, rnn_batch_filter: bool = False):
    """Silence third-party noise during ONNX export (torch.onnx / huggingface_hub).

    Spec-sanctioned: export.md §5.  FutureWarning is scoped to the three noisy
    libraries so teia's own FutureWarnings stay visible.
    """
    noisy = ["torch.onnx", "timm", "huggingface_hub"]
    saved = {n: logging.getLogger(n).level for n in noisy}
    for n in noisy:
        logging.getLogger(n).setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"(torch\.onnx|timm|huggingface_hub)(\..+)?")
            warnings.filterwarnings("ignore", module=r"huggingface_hub(\..+)?")
            warnings.filterwarnings("ignore", module=r"timm(\..+)?")
            if rnn_batch_filter:
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"Exporting a model to ONNX with a batch_size other than 1, "
                        r"with a variable length with (GRU|LSTM|RNN).*"
                    ),
                    category=UserWarning,
                    module=r"torch\.onnx\._internal\.torchscript_exporter\.symbolic_opset9",
                )
            yield
    finally:
        for n, level in saved.items():
            logging.getLogger(n).setLevel(level)


@contextmanager
def _disable_mha_fastpath_for_onnx_export(torch_module: Any):
    """Force ``nn.TransformerEncoderLayer`` to trace into standard ops during ONNX export.

    Under ``torch.no_grad()``, torch can swap it into a fused fast-path op with no ONNX
    symbolic; disabling the fast-path for the export's duration keeps tracing within the
    stable operator set. No-ops (and restores nothing) if this torch build lacks the flag.
    """
    mha = getattr(getattr(torch_module, "backends", None), "mha", None)
    get_enabled = getattr(mha, "get_fastpath_enabled", None)
    set_enabled = getattr(mha, "set_fastpath_enabled", None)
    if not callable(get_enabled) or not callable(set_enabled):
        yield
        return

    previous = bool(get_enabled())
    set_enabled(False)
    try:
        yield
    finally:
        set_enabled(previous)


def export_bundle(
    model: Any,
    config: dict[str, Any],
    run_dir: Path,
    checkpoint: str | None = None,
    output_dir: Path | None = None,
    datamodule: Any | None = None,
) -> Path:
    export_cfg = config.get("export") or {}
    layout_cfg = config.get("__runtime_layout__") or {}
    export_dir = ensure_dir(output_dir or (Path(layout_cfg.get("run_dir", run_dir)) / "export"))

    if hasattr(model, "export_teia_bundle"):
        return Path(
            _call_export_hook(
                model.export_teia_bundle,
                config=config,
                run_dir=run_dir,
                checkpoint=checkpoint,
                output_dir=export_dir,
                datamodule=datamodule,
            )
        )

    if hasattr(model, "to_onnx_bundle"):
        return Path(
            _call_export_hook(
                model.to_onnx_bundle,
                config=config,
                run_dir=run_dir,
                checkpoint=checkpoint,
                output_dir=export_dir,
                datamodule=datamodule,
            )
        )

    return _export_generic_onnx_bundle(
        model=model,
        config=config,
        datamodule=datamodule,
        checkpoint=checkpoint,
        export_cfg=export_cfg,
        export_dir=export_dir,
    )


def finalize_bundle(paths: BundlePaths, contract: BundleContract, model: Any, datamodule: Any | None = None) -> None:
    write_bundle_spec(paths, contract)
    write_bundle_runtime(paths, contract, model, datamodule)
    write_bundle_readme(paths, contract)


def _export_generic_onnx_bundle(
    *,
    model: Any,
    config: dict[str, Any],
    datamodule: Any | None,
    checkpoint: str | None,
    export_cfg: dict[str, Any],
    export_dir: Path,
) -> Path:
    torch = _require_torch_for_export()
    if datamodule is None:
        raise RuntimeError("Generic export requires an instantiated datamodule so task metadata can be resolved.")

    with suppress_export_noise(rnn_batch_filter=_has_rnn_modules(model, torch)):
        _configure_model_for_export(model=model, datamodule=datamodule)
        _restore_model_weights(model=model, checkpoint=checkpoint, torch_module=torch)
        contract = _resolve_generic_contract(
            model=model,
            config=config,
            datamodule=datamodule,
            model_relpath=f"onnx/{str(export_cfg.get('bundle_name') or 'model')}.onnx",
            torch_module=torch,
        )
        paths = BundlePaths.create(export_dir)
        model_path = paths.root / contract.model_relpath
        ensure_dir(model_path.parent)
        _export_model_onnx(model=model, contract=contract, model_path=model_path, torch_module=torch, config=config)
        finalize_bundle(paths, contract, model, datamodule)

    teia_log("export bundle complete", task=contract.task, bundle=paths.root, model=model_path)
    return paths.root


def _resolve_export_routes(model: Any, config: dict[str, Any]) -> list[str] | None:
    """Activation-route names to include (``export.routes`` config), default all."""
    configured = (config.get("export") or {}).get("routes")
    if configured is not None:
        return [str(name) for name in configured]
    available = [rec.name for rec in getattr(model, "_activation_records", []) or []]
    return available or None


def _resolve_field_specs(
    *, datamodule: Any, fields: list[str], torch_module: Any
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Per model-input field: ``(shape-without-batch-dim, dtype-name)``.

    Read off a real sample batch **after** ``on_after_batch_transfer`` (the batch-stage
    post-transfer rewrites, e.g. uint8→float normalize, are the last step before ``forward``
    sees the batch — so this is what the traced graph's declared input dtype/shape must match),
    or ``batch_meta()`` shapes with a float32 default for the interactive regime (interactive
    observations are always float and have no post-transfer stage).
    """
    if getattr(datamodule, "_live", False):
        meta = datamodule.batch_meta()
        return {name: (tuple(int(d) for d in meta.get(name, ())), "float32") for name in fields}
    sample = datamodule.on_after_batch_transfer(datamodule._sample_batch())
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for name in fields:
        value = getattr(sample, name)
        dtype_name = str(getattr(value, "dtype", torch_module.float32)).rsplit(".", 1)[-1]
        specs[name] = (tuple(int(d) for d in value.shape[1:]), dtype_name)
    return specs


def _resolve_inputs(
    *, datamodule: Any, fields: list[str], field_specs: dict[str, tuple[tuple[int, ...], str]]
) -> tuple[list[InputSpec], dict[str, list[PreprocessStep]]]:
    resolve_chain = resolve_chain_for(datamodule)
    inputs: list[InputSpec] = []
    chains: dict[str, list[PreprocessStep]] = {}
    for name in fields:
        shape, dtype = field_specs[name]
        chain, payload_key, payload_spec = resolve_chain(datamodule, name)
        inputs.append(
            InputSpec(name=name, payload_key=payload_key, dtype=dtype, shape=["batch", *shape], payload=payload_spec)
        )
        chains[name] = chain
    return inputs, chains


def _collect_export_artifacts(datamodule: Any) -> dict[str, dict[str, Any]]:
    """Merge every data node's ``export_artifacts()`` (empty for the interactive regime, which has
    no data-node graph)."""
    if getattr(datamodule, "_live", False):
        return {}
    extra: dict[str, dict[str, Any]] = {}
    for node in datamodule._nodes.values():
        extra.update(node.export_artifacts())
    return extra


def _resolve_generic_contract(
    *,
    model: Any,
    config: dict[str, Any],
    datamodule: Any,
    model_relpath: str,
    torch_module: Any = None,
) -> BundleContract:
    torch_module = torch_module or _require_torch_for_export()
    task_name = (config.get("task") or {}).get("_target_")
    fields = resolve_forward_batch_fields(model)
    if not fields:
        raise RuntimeError(
            f"Generic ONNX export found no batch.* inputs on {type(model).__name__!r}'s pipeline records; "
            "provide export_teia_bundle() on the model instead."
        )
    routes = _resolve_export_routes(model, config)
    field_specs = _resolve_field_specs(datamodule=datamodule, fields=fields, torch_module=torch_module)
    inputs, preprocess_chains = _resolve_inputs(datamodule=datamodule, fields=fields, field_specs=field_specs)

    outputs, activation_outputs, variable_axes = _discover_outputs(
        model=model, fields=fields, field_specs=field_specs, torch_module=torch_module, routes=routes
    )
    labels = [str(name) for name in datamodule.meta().get("class_names") or []]
    postprocess: dict[str, Any] = {"task": task_name, "activation_outputs": activation_outputs}
    if activation_outputs:
        postprocess.setdefault("conf", 0.25)
        postprocess.setdefault("iou", 0.45)
        postprocess.setdefault("max_det", 300)

    dynamic_axes: dict[str, dict[int, str]] = {item.name: {0: "batch"} for item in inputs}
    for name in [item.get("name") for item in outputs if item.get("name")]:
        dynamic_axes[str(name)] = {0: "batch"}
    for name, axes in variable_axes.items():
        if name in dynamic_axes:
            dynamic_axes[name].update(axes)

    return BundleContract(
        task=task_name,
        exporter_family="teia.generic",
        model_relpath=model_relpath,
        inputs=inputs,
        preprocess_chains=preprocess_chains,
        labels=list(labels),
        routes=routes,
        outputs=outputs,
        dynamic_axes=dynamic_axes,
        postprocess=postprocess,
        extra_metadata=_collect_export_artifacts(datamodule),
    )


def _configure_model_for_export(*, model: Any, datamodule: Any) -> None:
    ensure_metadata = getattr(datamodule, "ensure_task_metadata", None)
    if callable(ensure_metadata):
        ensure_metadata()
    if hasattr(model, "configure_from_datamodule"):
        model.configure_from_datamodule(datamodule)
    if hasattr(model, "eval"):
        model.eval()
    modules_fn = getattr(model, "modules", None)
    if callable(modules_fn):
        for submodule in modules_fn():
            configure_for_export = getattr(submodule, "configure_for_export", None)
            if callable(configure_for_export):
                configure_for_export()


def _restore_model_weights(*, model: Any, checkpoint: str | None, torch_module: Any) -> None:
    if not checkpoint:
        return
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu", torch_module=torch_module)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    loader = getattr(model, "load_state_dict", None)
    if not callable(loader):
        raise TypeError(f"Configured model {type(model)!r} does not support load_state_dict().")
    result = loader(state_dict, strict=False)
    if getattr(result, "unexpected_keys", None):
        warnings.warn(
            f"Export: checkpoint keys not loaded into the export model (unexpected): "
            f"{result.unexpected_keys}. The exported model may differ from what was trained.",
            UserWarning,
            stacklevel=2,
        )
    if getattr(result, "missing_keys", None):
        warnings.warn(
            f"Export: model parameters not found in checkpoint (missing): "
            f"{result.missing_keys}. These will use random initialization.",
            UserWarning,
            stacklevel=2,
        )


def _export_model_onnx(
    *,
    model: Any,
    contract: BundleContract,
    model_path: Path,
    torch_module: Any,
    config: dict[str, Any],
) -> None:
    """Trace ``contract``'s wrapper through dynamo to ONNX.

    ``torch.export`` specializes size-0/1 dims as constants, so a dynamic batch axis needs an
    example extent >= 2 (core/export.md); RNN modules trace most portably at batch_size=1, so
    that case keeps the batch axis static instead of dynamic.
    """
    static_batch = _has_rnn_modules(model, torch_module)
    batch_extent = 1 if static_batch else 2
    dummy_inputs = tuple(
        torch_module.zeros([batch_extent, *[int(d) for d in item.shape[1:]]], dtype=getattr(torch_module, item.dtype))
        for item in contract.inputs
    )

    export_target = _build_export_wrapper(
        model=model, fields=[item.name for item in contract.inputs], routes=contract.routes
    )
    input_names = [item.name for item in contract.inputs]
    output_names = [str(item.get("name")) for item in contract.outputs if item.get("name")]

    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": int((config.get("export") or {}).get("opset", 18)),
        "dynamo": True,
    }
    if not static_batch:
        export_kwargs["dynamic_shapes"] = [{0: torch_module.export.Dim("batch")} for _ in contract.inputs]

    teia_log(
        "export.onnx start",
        model_path=model_path,
        input_names=input_names,
        output_names=output_names,
        opset=export_kwargs["opset_version"],
        static_batch=static_batch,
    )
    with _disable_mha_fastpath_for_onnx_export(torch_module):
        torch_module.onnx.export(export_target, dummy_inputs, str(model_path), **export_kwargs)
    teia_log("export.onnx complete", model_path=model_path)


def _strip_non_tensors(value: Any, tensor_type: type) -> Any:
    """Recursively extract tensor leaves from NamedTuples/tuples/lists, dropping non-tensors."""
    if isinstance(value, tensor_type):
        return value
    if hasattr(value, "_fields") or isinstance(value, (tuple, list)):
        children = list(value)
        tensors: list[Any] = []
        for item in children:
            extracted = _strip_non_tensors(item, tensor_type)
            if isinstance(extracted, tensor_type):
                tensors.append(extracted)
            elif isinstance(extracted, tuple):
                tensors.extend(extracted)
        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)
    return value


def _flatten_export_outputs(
    model: Any, result: Any, tensor_type: type, *, routes: list[str] | None = None
) -> list[tuple[str, Any]]:
    """Flatten the traced graph outputs as ``(name, tensor)`` pairs. See ``_flatten_export_outputs_named``
    for the variant that also reports which names are activation (vs. raw logit) outputs."""
    return [(name, tensor) for name, tensor, _is_activation in _flatten_export_outputs_named(model, result, tensor_type, routes=routes)]


def _flatten_export_outputs_named(
    model: Any, result: Any, tensor_type: type, *, routes: list[str] | None = None
) -> list[tuple[str, Any, bool]]:
    """Flatten the traced graph outputs as ``(name, tensor, is_activation)`` triples.

    The exported ONNX graph carries **both** the raw logits **and** the activated outputs for
    each selected route (``routes=None`` selects all — teia:core/module/activation_route.md,
    teia:core/export.md): the logit field names of the ``Pred`` NamedTuple plus the keys of
    each selected loss ``activation`` dict. Non-tensor leaves (bound methods, dicts) are
    dropped — for a detector the raw logit field is a non-tensor dict, so only the decoded
    activation tensors remain.

    Activation outputs are added first so they claim their un-suffixed key names — the bundle's
    ROUTES mapping (assemble.py) always looks up a route's activation-dict keys verbatim, with
    no knowledge of collision renames. A raw Pred field that happens to share a name with an
    activation-dict key (e.g. a decoder field also called "reconstructed") must be the one
    that gets bumped to ``_1``, not the activated tensor ROUTES expects to find — so
    ``is_activation`` reports the true origin per output name, not a name-membership guess.
    """
    pairs: list[tuple[str, Any, bool]] = []
    used: set[str] = set()

    def add(name: str, tensor: Any, is_activation: bool) -> None:
        candidate = name
        suffix = 1
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate)
        pairs.append((candidate, tensor, is_activation))

    activate = getattr(model, "activate", None)
    if callable(activate):
        activated = activate(result)
        if isinstance(activated, dict):
            for route_name, alias_outputs in activated.items():
                if routes is not None and route_name not in routes:
                    continue
                if isinstance(alias_outputs, dict):
                    for key, tensor in alias_outputs.items():
                        if isinstance(tensor, tensor_type):
                            add(key, tensor, True)

    fields = getattr(result, "_fields", None)
    if fields:
        for field_name in fields:
            extracted = _strip_non_tensors(getattr(result, field_name), tensor_type)
            if isinstance(extracted, tensor_type):
                add(field_name, extracted, False)
            elif isinstance(extracted, tuple):
                for index, item in enumerate(extracted):
                    if isinstance(item, tensor_type):
                        add(f"{field_name}_{index}", item, False)
    else:
        extracted = _strip_non_tensors(result, tensor_type)
        if isinstance(extracted, tensor_type):
            add("output", extracted, False)
        elif isinstance(extracted, tuple):
            for index, item in enumerate(extracted):
                if isinstance(item, tensor_type):
                    add(f"output_{index}", item, False)

    return pairs


def _discover_outputs(
    *,
    model: Any,
    fields: list[str],
    field_specs: dict[str, tuple[tuple[int, ...], str]],
    torch_module: Any,
    routes: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, dict[int, str]]]:
    """Run a dummy ``forward + activation`` to enumerate ONNX output names/shapes.

    Returns ``(outputs, activation_output_names, variable_axes)`` — the third merges each selected
    activation's ``DYNAMIC_AXES`` (activated key → ``{axis: name}``) for the ONNX dynamic axes.
    """
    forward = getattr(model, "forward", None)
    if not callable(forward):
        return ([{"name": "logits", "dtype": "float32", "shape": ["batch", "..."]}], ["logits"], {})

    dummy_fields = {
        name: torch_module.zeros((1, *field_specs[name][0]), dtype=getattr(torch_module, field_specs[name][1]))
        for name in fields
    }
    dummy = SimpleNamespace(**dummy_fields)
    with torch_module.no_grad():
        pred = forward(dummy)

    pairs = _flatten_export_outputs_named(model, pred, torch_module.Tensor, routes=routes)
    if not pairs:
        return ([{"name": "logits", "dtype": "float32", "shape": ["batch", "..."]}], ["logits"], {})

    # The ONNX graph carries both the raw Pred logit fields and the activation outputs, but the
    # bundle's decode consumes only the loss-owned activation dict (what `model.activate` returns).
    # So `activation_outputs` excludes the raw Pred fields; their ONNX outputs still exist. Uses
    # each pair's true origin (`is_activation`), not a name-membership guess, since a raw field
    # can collide-rename with an activation key (or vice versa) and still needs the right label.
    outputs: list[dict[str, Any]] = []
    activation_outputs: list[str] = []
    for name, tensor, is_activation in pairs:
        shape: list[Any] = ["batch"] + [int(dim) for dim in tuple(tensor.shape)[1:]]
        outputs.append({"name": name, "dtype": "float32", "shape": shape})
        if is_activation:
            activation_outputs.append(name)

    variable_axes: dict[str, dict[int, str]] = {}
    for rec in model._activation_records:
        if routes is None or rec.name in routes:
            variable_axes.update(type(getattr(model, sanitize_key(rec.name))).DYNAMIC_AXES)
    return outputs, activation_outputs, variable_axes


def _build_export_wrapper(*, model: Any, fields: list[str], routes: list[str] | None) -> Any:
    """A fixed-arity ``nn.Module`` wrapper: one positional argument per model-input field.

    Traces ``forward`` then the loss-owned ``activation`` for the selected routes, emitting
    ``(*logits, *activated)`` (teia:core/module/activation_route.md §5). The batch object passed
    to ``forward`` is a duck-typed namespace over exactly the fields the generated forward reads
    (attribute access only — no NamedTuple identity check — teia:core/module/codegen.md).
    """
    import torch
    import torch.nn as nn

    tensor_type = torch.Tensor

    class _ExportWrapper(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def run(self, provided: dict[str, Any]) -> Any:
            result = self.inner.forward(SimpleNamespace(**provided))
            pairs = _flatten_export_outputs(self.inner, result, tensor_type, routes=routes)
            if not pairs:
                return _strip_non_tensors(result, tensor_type)
            tensors = [tensor for _, tensor in pairs]
            return tensors[0] if len(tensors) == 1 else tuple(tensors)

    wrapper = _ExportWrapper(model)
    params = ", ".join(fields)
    mapping = ", ".join(f"{name!r}: {name}" for name in fields)
    source = f"def forward(self, {params}):\n    return self.run({{{mapping}}})\n"
    namespace: dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 - fixed-arity signature for ONNX named-input registration
    wrapper.forward = namespace["forward"].__get__(wrapper, _ExportWrapper)
    wrapper.eval()
    return wrapper


def _has_rnn_modules(model: Any, torch_module: Any) -> bool:
    rnn_base = getattr(getattr(getattr(torch_module, "nn", None), "modules", None), "rnn", None)
    rnn_base = getattr(rnn_base, "RNNBase", None)
    if rnn_base is None:
        return False
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        return False
    return any(isinstance(module, rnn_base) for module in modules_fn())


def _call_export_hook(
    hook: Any,
    *,
    config: dict[str, Any],
    run_dir: Path,
    checkpoint: str | None,
    output_dir: Path,
    datamodule: Any | None,
) -> str | Path:
    kwargs = {
        "config": config,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "datamodule": datamodule,
    }
    try:
        parameters = inspect.signature(hook).parameters
    except (TypeError, ValueError):
        parameters = {}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        accepted = dict(kwargs)
    else:
        accepted = {name: value for name, value in kwargs.items() if name in parameters}
    return hook(**accepted)


def _require_torch_for_export() -> Any:
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Teia export requires torch/onnx support unless the model supplies export_teia_bundle()."
        ) from exc
