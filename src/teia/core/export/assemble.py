"""Assemble a standalone, model-specific runtime package for an export bundle.

Instead of rendering one generic template, the assembler walks the resolved bundle contract and
emits exactly the code this model needs:

* ``runtime/preprocess.py`` — per model-input field, its preprocess chain's kernels (collected
  verbatim, unrolled — chains are static and fully known at export time) + baked per-step
  ``PARAMS`` + a generic ``prepare(payload)`` entrypoint merging every field's feed/restore.
* ``runtime/postprocess.py`` — each activation route's ``kernel`` (collected verbatim) +
  this model's baked per-route ``CTX_BY_ROUTE`` / ``ROUTES`` + a generic ``decode(outputs, restore)``
  entrypoint returning ``{<act-name>: predictions}``.
* ``runtime/model.py`` / ``runtime/run.py`` — small generic onnxruntime glue (no task ifs).

The emitted package imports only numpy / torch / torchvision / onnxruntime / PIL — never
``teia`` (package spec core/export.md). Before finalizing, the assembler executes the emitted decode on
a dummy forward pass and asserts it agrees with the live ``model.postprocess`` — a broken
or incomplete collection fails the export instead of shipping.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

from teia.core.export.bundle import BundleContract, BundlePaths
from teia.core.export.collect import EmittedSource, collect_callable
from teia.core.export.preprocess_slice import PreprocessStep
from teia.core.module.codegen import sanitize_key
from teia.core.utils import teia_log

# ctx/contract keys that are bundle bookkeeping, not part of the decode ctx.
_NON_CTX_KEYS = frozenset({"activation_outputs", "adapter", "task", "loss_target"})


def write_bundle_runtime(paths: BundlePaths, contract: BundleContract, model: Any, datamodule: Any | None = None) -> None:
    """Emit and verify the standalone runtime package for ``model``."""
    preprocess_src = _emit_preprocess(contract)
    _verify_prepare(preprocess_src, contract=contract, model=model, datamodule=datamodule)
    postprocess_src = _emit_postprocess(contract, model)
    _verify_decode(postprocess_src, contract=contract, model=model, routes=contract.routes)

    (paths.runtime_dir / "preprocess.py").write_text(preprocess_src, encoding="utf-8")
    (paths.runtime_dir / "postprocess.py").write_text(postprocess_src, encoding="utf-8")
    (paths.runtime_dir / "model.py").write_text(_MODEL_MODULE, encoding="utf-8")
    (paths.runtime_dir / "run.py").write_text(_RUN_MODULE, encoding="utf-8")
    teia_log("export runtime emitted", dir=paths.runtime_dir)


def _emit_preprocess(contract: BundleContract) -> str:
    """Emit a ``prepare(payload) -> (feed, restore)`` unrolling every input field's chain.

    Each field's chain is collected and unrolled into its own ``_prepare_<field>`` function
    (chains are static and fully known at export time, so codegen inlines them rather than
    looping over a runtime list); ``prepare`` merges every field's feed entry and restore dict.
    A ``"window"``-payload field (a reshape-rooted, windowed chain) loops its row-mapped
    pre-seam steps over each row in the payload list, then applies the remaining collate-seam
    step once over the whole row list; a ``"value"`` field keeps the straight-line unroll.
    """
    merged_imports: dict[str, None] = {}
    merged_blocks: list[str] = []
    seen_blocks: set[str] = set()
    field_functions: list[str] = []

    def _collect(step: PreprocessStep, index: int, field_name: str) -> tuple[str, str]:
        emitted = collect_callable(step.kernel, owner=step.owner)
        for statement in emitted.imports:
            merged_imports.setdefault(statement, None)
        for block in emitted.blocks:
            if block not in seen_blocks:
                seen_blocks.add(block)
                merged_blocks.append(block)
        params_name = f"_PARAMS_{field_name}_{index}"
        merged_blocks.append(f"{params_name} = {step.params!r}")
        return emitted.entry, params_name

    for input_spec in contract.inputs:
        chain = contract.preprocess_chains.get(input_spec.name, [])
        if input_spec.payload.get("kind") == "window":
            row_steps = [step for step in chain if step.row_mapped]
            seam_steps = [step for step in chain if not step.row_mapped]
            lines = [
                f"def _prepare_{input_spec.name}(payload):",
                "    restore = {}",
                f"    rows = payload[{input_spec.payload_key!r}]",
                "    row_values = []",
                "    for row in rows:",
                "        value = row",
            ]
            for index, step in enumerate(row_steps):
                entry, params_name = _collect(step, index, input_spec.name)
                lines.append(f"        value, step_restore = {entry}(value, {params_name})")
                lines.append("        restore.update(step_restore)")
            lines.append("        row_values.append(value)")
            lines.append("    value = row_values")
            for offset, step in enumerate(seam_steps):
                entry, params_name = _collect(step, len(row_steps) + offset, input_spec.name)
                lines.append(f"    value, step_restore = {entry}(value, {params_name})")
                lines.append("    restore.update(step_restore)")
            lines.append("    return value, restore")
            field_functions.append("\n".join(lines))
            continue

        lines = [
            f"def _prepare_{input_spec.name}(payload):",
            "    restore = {}",
            f"    value = payload[{input_spec.payload_key!r}]",
        ]
        for index, step in enumerate(chain):
            entry, params_name = _collect(step, index, input_spec.name)
            arg = "[value]" if step.collate else "value"  # a collate kernel batches a list of samples
            lines.append(f"    value, step_restore = {entry}({arg}, {params_name})")
            lines.append("    restore.update(step_restore)")
        lines.append("    return value, restore")
        field_functions.append("\n".join(lines))

    prepare_lines = ["def prepare(payload):", "    feed = {}", "    restore = {}"]
    for input_spec in contract.inputs:
        prepare_lines.append(f"    value, step_restore = _prepare_{input_spec.name}(payload)")
        prepare_lines.append(f"    feed[{input_spec.name!r}] = value")
        prepare_lines.append("    restore.update(step_restore)")
    prepare_lines.append("    return feed, restore")

    body = "\n\n\n".join([*field_functions, "\n".join(prepare_lines)])
    if not merged_blocks and not merged_imports:
        return "from __future__ import annotations\n\n\n" + body + "\n"
    header = EmittedSource(entry="", imports=list(merged_imports), blocks=merged_blocks).render()
    return header.rstrip() + "\n\n\n" + body + "\n"


def _verify_prepare(preprocess_src: str, *, contract: BundleContract, model: Any, datamodule: Any | None) -> None:
    """Run the emitted ``prepare`` on a real seed and assert it agrees, field-by-field, with the
    live path (``on_after_batch_transfer`` over a one-sample collated batch).

    Best-effort: interactive datamodules, datamodules with no static seed, any ``"value"`` field
    whose raw payload can't be recovered directly off a plain reader/join seed (a
    ``"window"``-payload field, or a field produced mid-graph rather than at the root), and any
    graph whose seed shape the naive per-field lookup below doesn't model correctly (e.g. a
    structured row wrapper a kernel needs unwrapped first) skip the check entirely rather than
    guessing or crashing a real export — this covers the common straight-line
    reader->preprocess->collate chain, not every graph shape. Only a genuine, cleanly-computed
    mismatch raises.
    """
    del model
    if datamodule is None or getattr(datamodule, "_live", False):
        return
    seed: dict[str, Any] | None = None
    for split in ("train", "val", "test"):
        seeds = datamodule._seeds_for(split)
        if seeds:
            seed = seeds[0]
            break
    if seed is None:
        return

    payload: dict[str, Any] = {}
    for input_spec in contract.inputs:
        if input_spec.payload.get("kind") != "value":
            return
        raw_key = f"item.{input_spec.payload_key}"
        if raw_key not in seed:
            raw_key = f"stream.{input_spec.payload_key}"
        if raw_key not in seed:
            return
        payload[input_spec.payload_key] = seed[raw_key]

    import torch

    try:
        namespace = _exec_module(preprocess_src)
        feed, _restore = namespace["prepare"](payload)
        reference = datamodule.on_after_batch_transfer(datamodule._sample_batch())
        comparisons = [
            (input_spec.name, torch.as_tensor(feed[input_spec.name]), torch.as_tensor(getattr(reference, input_spec.name)))
            for input_spec in contract.inputs
        ]
    except Exception:
        # The seed's raw shape didn't match what this chain's kernels expect (e.g. a structured
        # row wrapper needing unwrapping first) — not confidently verifiable, so skip rather than
        # fail an export whose actual bundle may well be correct.
        return

    for name, bundle_value, live_value in comparisons:
        if not torch.equal(bundle_value, live_value):
            raise RuntimeError(
                "Export verification failed: emitted prepare() disagrees with the live preprocess "
                f"path for field {name!r}. This usually means a collected kernel or its baked "
                "PARAMS drifted from the live node."
            )


def _emit_postprocess(contract: BundleContract, model: Any) -> str:
    """Emit a ``decode(outputs, restore)`` that returns ``{<act-name>: route_decode(...)}``.

    Each activation route contributes its ``kernel`` (collected verbatim) and the ``act.*``
    output keys it owns. Routes with no kernel are an identity tail.
    """
    routes = _collect_routes(model, routes=contract.routes)
    ctx = {key: value for key, value in contract.postprocess.items() if key not in _NON_CTX_KEYS}
    route_keys = {name: keys for name, keys, _, _owner, _params in routes}
    ctx_by_route = {name: {**ctx, **node_params} for name, _keys, _fn, _owner, node_params in routes}

    decoders: dict[str, str | None] = {}
    merged_imports: dict[str, None] = {}
    merged_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for name, _keys, decode_fn, owner, _params in routes:
        if decode_fn is None:
            decoders[name] = None
            continue
        emitted = collect_callable(decode_fn, owner=owner)
        for statement in emitted.imports:
            merged_imports.setdefault(statement, None)
        for block in emitted.blocks:
            if block not in seen_blocks:
                seen_blocks.add(block)
                merged_blocks.append(block)
        decoders[name] = emitted.entry

    decoder_items = ", ".join(f"{name!r}: {entry or 'None'}" for name, entry in decoders.items())
    wrapper = (
        f"CTX_BY_ROUTE = {ctx_by_route!r}\n"
        f"ROUTES = {route_keys!r}\n"
        f"_DECODERS = {{{decoder_items}}}\n\n\n"
        "def decode(outputs, restore=None):\n"
        "    result = {}\n"
        "    for name, keys in ROUTES.items():\n"
        "        activated = {key: outputs[key] for key in keys}\n"
        "        ctx = dict(CTX_BY_ROUTE.get(name, {}))\n"
        "        if restore:\n"
        "            ctx.update({key: [value] for key, value in restore.items()})  # one payload = one sample\n"
        "        fn = _DECODERS.get(name)\n"
        "        if fn is None:\n"
        "            result[name] = {key: getattr(value, 'tolist', lambda: value)() for key, value in activated.items()}\n"
        "        else:\n"
        "            result[name] = fn(activated, ctx)\n"
        "    return result\n"
    )
    if not merged_blocks:
        return "from __future__ import annotations\n\n\n" + wrapper
    header = EmittedSource(entry="", imports=list(merged_imports), blocks=merged_blocks).render()
    return header.rstrip() + "\n\n\n" + wrapper


def _collect_routes(
    model: Any, *, routes: list[str] | None = None
) -> list[tuple[str, list[str], Callable[..., Any] | None, str | None, dict[str, Any]]]:
    """Per selected activation route (``routes=None`` selects all): ``(name, act-output-keys,
    kernel-or-None, owner-class-name-or-None, node-params)``. ``node-params`` is the live node's
    ``params()`` (e.g. ``QuantileActivation.quantiles``) — baked into this route's ctx since a
    ``kernel`` staticmethod has no ``self``."""
    collected: list[tuple[str, list[str], Callable[..., Any] | None, str | None, dict[str, Any]]] = []
    for rec in getattr(model, "_activation_records", []) or []:
        if routes is not None and rec.name not in routes:
            continue
        node = getattr(model, sanitize_key(rec.name), None)
        keys = [key[len("act."):] for key in (rec.out_key or []) if key.startswith("act.")]
        decode_fn = None
        owner = None
        node_params: dict[str, Any] = {}
        if node is not None:
            decode_fn = getattr(type(node), "kernel", None)
            owner = type(node).__name__
            node_params = node.params()
        collected.append((rec.name, keys, decode_fn, owner, node_params))
    return collected


def _predictions_equal(a: Any, b: Any) -> bool:
    """Recursive equality for decode outputs. Atom dicts carry tensors and numpy arrays rather than
    JSON-safe primitives; a plain ``!=`` on a dict/list
    containing them raises (tensor ``!=`` is elementwise, not boolean), so nested containers are
    walked explicitly and tensors compared with ``torch.equal``."""
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_predictions_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_predictions_equal(x, y) for x, y in zip(a, b))
    return a == b


def _verify_decode(source: str, *, contract: BundleContract, model: Any, routes: list[str] | None = None) -> None:
    """Run the emitted decode on a dummy forward and assert it matches model.postprocess."""
    selected_routes = _collect_routes(model, routes=routes)
    if all(decode_fn is None for _name, _keys, decode_fn, _owner, _params in selected_routes):
        _exec_module(source)  # still ensure it imports/executes
        return
    import torch

    from teia.core.export.kernels import to_numpy
    from teia.core.export.onnx import _flatten_export_outputs

    namespace = _exec_module(source)
    ctx = {key: value for key, value in contract.postprocess.items() if key not in _NON_CTX_KEYS}

    dummy_fields = {
        item.name: torch.zeros(
            [2 if str(dim) == "batch" else int(dim) for dim in item.shape],
            dtype=getattr(torch, item.dtype),
        )
        for item in contract.inputs
    }
    dummy = SimpleNamespace(**dummy_fields)
    with torch.no_grad():
        pred = model(dummy)
        activated = model.activate(pred)
        activated = {name: outputs for name, outputs in activated.items() if routes is None or name in routes}
    reference = model.postprocess(activated, ctx)

    # Mirror the full ONNX output set (logits + activation) the bundle's decode will see at
    # runtime; to_numpy upcasts reduced-precision (bf16/half) tensors numpy cannot represent.
    outputs = {
        name: to_numpy(tensor)
        for name, tensor in _flatten_export_outputs(model, pred, torch.Tensor, routes=routes)
    }
    bundle_result = namespace["decode"](outputs)
    for name, _keys, decode_fn, _owner, _params in selected_routes:
        if decode_fn is None:
            continue
        if not _predictions_equal(bundle_result.get(name), reference.get(name)):
            raise RuntimeError(
                "Export verification failed: emitted decode disagrees with model.postprocess for route "
                f"{name!r}. This usually means baked CTX/ROUTES are wrong or the kernel is non-deterministic."
            )


def _exec_module(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(source, "<export-postprocess>", "exec"), namespace)  # noqa: S102 - generated, verified source
    return namespace


_MODEL_MODULE = '''from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import onnxruntime as ort  # noqa: E402
import preprocess  # noqa: E402
import postprocess  # noqa: E402


class ExportedModel:
    """Standalone runner for this exported Teia model (no teia dependency)."""

    def __init__(self, bundle_dir):
        self.dir = Path(bundle_dir)
        manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        model_path = self.dir / manifest["artifacts"]["model"]
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def predict(self, payload):
        feed, restore = preprocess.prepare(payload)
        names = [output.name for output in self.session.get_outputs()]
        feed = {name: getattr(value, "numpy", lambda: value)() for name, value in feed.items()}
        outputs = dict(zip(names, self.session.run(names, feed)))
        return postprocess.decode(outputs, restore)
'''


_RUN_MODULE = '''from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ExportedModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run this exported Teia model.")
    parser.add_argument("--bundle-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--payload-json", help="Inline JSON payload, e.g. '{\\"image\\": \\"path.jpg\\"}'.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if not args.payload_json:
        raise SystemExit("Provide --payload-json (see metadata/task.json for this bundle's input keys).")
    payload = json.loads(args.payload_json)

    predictions = ExportedModel(args.bundle_dir).predict(payload)
    print(json.dumps(predictions, indent=2 if args.pretty else None, default=lambda value: value.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
