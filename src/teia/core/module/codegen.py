from __future__ import annotations

import types

# sanitize_key's impl moved to teia.base.keys (a component callback needs it); re-exported here.
from teia.base.keys import sanitize_key


def key_to_expr(key: str) -> str:
    """Convert a pipeline key to the Python expression used in generated code.

    ``batch.X``  → ``batch.X``   (attribute access on the batch NamedTuple)
    ``feat.X``   → ``feat_X``    (local variable)
    ``pred.X``   → ``pred_X``    (local variable; later packed into Pred)
    """
    if key.startswith("batch."):
        return key
    return sanitize_key(key)


def validate_pipeline(records: list, loss_records: list, activation_records: list | None = None) -> None:
    """Raise ValueError if the pipeline graph has structural problems."""
    defined: set[str] = set()

    def is_available(k: str) -> bool:
        return k.startswith("batch.") or k in defined

    for rec in records:
        for k in rec.in_key:
            if not is_available(k):
                raise ValueError(
                    f"Node '{rec.name}': in_key '{k}' is not defined by any "
                    f"preceding node and is not a batch.* key."
                )
        if rec.out_key:
            for k in rec.out_key:
                if k in defined:
                    raise ValueError(f"Duplicate out_key '{k}' in pipeline.")
                defined.add(k)

    loss_defined: set[str] = set()
    for rec in loss_records:
        for k in rec.in_key:
            if not (k.startswith("batch.") or k.startswith("pred.")):
                raise ValueError(
                    f"Loss node '{rec.name}': in_key '{k}' must be batch.* or pred.*; "
                    f"feat.* keys are not accessible in loss computation."
                )
        # aux_in is the sole sanctioned cross-loss read: it may reference loss.* terms from an
        # earlier loss node (gradient-norm balancing).
        for k in getattr(rec, "aux_in", []) or []:
            if not k.startswith("loss."):
                raise ValueError(
                    f"Loss node '{rec.name}': aux_in '{k}' must reference a loss.* term."
                )
            if k not in loss_defined:
                raise ValueError(
                    f"Loss node '{rec.name}': aux_in '{k}' is not produced by an earlier loss node."
                )
        for k in rec.out_key:
            if not k.startswith("loss."):
                raise ValueError(
                    f"Loss node '{rec.name}': out_key '{k}' must start with 'loss.'."
                )
            if k in loss_defined:
                raise ValueError(f"Duplicate loss out_key '{k}'.")
            loss_defined.add(k)

    act_defined: set[str] = set()
    for rec in activation_records or []:
        for k in rec.in_key:
            if not k.startswith("pred."):
                raise ValueError(
                    f"Activation node '{rec.name}': in_key '{k}' must be pred.*; "
                    f"targets and feat.* keys never enter the prediction route."
                )
        for k in rec.out_key:
            if not k.startswith("act."):
                raise ValueError(
                    f"Activation node '{rec.name}': out_key '{k}' must start with 'act.'."
                )
            if k in act_defined:
                raise ValueError(f"Duplicate activation out_key '{k}'.")
            act_defined.add(k)


def _in_expr(k: str, detach: set[str]) -> str:
    """Forward in-key expression, stop-gradient'd when listed in ``node.detach``."""
    expr = key_to_expr(k)
    return f"{expr}.detach()" if k in detach else expr


def generate_forward_fn(records: list, pred_fields: list[str]) -> str:
    """Return source string for _generated_forward(self, batch) -> Pred.

    Nodes are **None-tolerant**: a node runs only if at least one of its inputs is non-None,
    otherwise its outputs are set to None.
    (see teia:core/module/codegen.md)."""
    lines = ["def _generated_forward(self, batch):"]
    for rec in records:
        # A 'tied' bind node has no module of its own — it re-applies the source node's module
        # (shared weights) at this call site. 'frozen'/own nodes call their own attribute.
        bind = getattr(rec, "bind", None)
        callee = bind.from_node if (bind is not None and bind.mode == "tied") else rec.name
        node_attr = f"self.{sanitize_key(callee)}"
        detach = set(getattr(rec, "detach", []) or [])
        args = ", ".join(_in_expr(k, detach) for k in rec.in_key)
        guard = " or ".join(f"{key_to_expr(k)} is not None" for k in rec.in_key)

        if not rec.out_key:
            if guard:
                lines.append(f"    if {guard}:")
                lines.append(f"        {node_attr}({args})")
            else:
                lines.append(f"    {node_attr}({args})")
            continue

        lhs = ", ".join(key_to_expr(k) for k in rec.out_key)
        if not guard:
            lines.append(f"    {lhs} = {node_attr}({args})")
        elif len(rec.out_key) == 1:
            lines.append(f"    {lhs} = {node_attr}({args}) if ({guard}) else None")
        else:
            none_tuple = "(" + ", ".join("None" for _ in rec.out_key) + ")"
            lines.append(f"    {lhs} = {node_attr}({args}) if ({guard}) else {none_tuple}")

    pred_args = ", ".join(key_to_expr(f) for f in pred_fields)
    lines.append(f"    return self._pred_type({pred_args})")
    return "\n".join(lines)


def _loss_in_expr(k: str, detach: set[str] | None = None) -> str:
    """Convert a loss node in_key to its expression in _generated_losses."""
    detach = detach or set()
    if k.startswith("batch."):
        expr = k
    elif k.startswith("pred."):
        expr = f"pred.{k[len('pred.'):]}"
    else:
        raise ValueError(f"Loss in_key '{k}' must be batch.* or pred.*")
    return f"{expr}.detach()" if k in detach else expr


def generate_losses_fn(loss_records: list) -> str:
    """Return source string for _generated_losses(self, batch, pred) -> dict.

    Each loss node declares one or more ``loss.*`` out_keys.  Single-output
    nodes return a scalar Tensor; multi-output nodes return a NamedTuple that
    is unpacked by position, mirroring how forward nodes handle multiple
    out_keys.  Every ``loss.*`` value is logged individually; the weighted sum
    across all nodes becomes ``loss_log['loss']``.
    """
    lines = ["def _generated_losses(self, batch, pred):"]
    lines.append("    total_loss = None")
    lines.append("    loss_log = {}")
    for rec in loss_records:
        node_attr = f"self.{sanitize_key(rec.name)}"
        detach = set(getattr(rec, "detach", []) or [])
        call_args = [_loss_in_expr(k, detach) for k in rec.in_key]
        # Adaptive-weight hook, appended after in_key args in (nll, last_layer) order: aux_in terms
        # (summed) -> the node's `nll`; last_layer -> a forward alias's reference weight tensor.
        aux_in = getattr(rec, "aux_in", []) or []
        if aux_in:
            call_args.append("(" + " + ".join(sanitize_key(k) for k in aux_in) + ")")
        last_layer = getattr(rec, "last_layer", None)
        if last_layer:
            call_args.append(f"self._resolve_last_layer({last_layer!r})")
        args = ", ".join(call_args)
        out_keys = rec.out_key  # list of loss.* keys

        if len(out_keys) == 1:
            var = sanitize_key(out_keys[0])
            weight = f"self._loss_term_weight['{out_keys[0]}']"
            lines.append(f"    {var} = {node_attr}({args})")
            lines.append(f"    loss_log['{out_keys[0]}'] = {var}")
            lines.append("    if total_loss is None:")
            lines.append(f"        total_loss = {weight} * {var}")
            lines.append("    else:")
            lines.append(f"        total_loss = total_loss + {weight} * {var}")
        else:
            lhs = ", ".join(sanitize_key(k) for k in out_keys)
            lines.append(f"    {lhs} = {node_attr}({args})")
            for k in out_keys:
                var = sanitize_key(k)
                lines.append(f"    loss_log['{k}'] = {var}")
            weighted_terms = [f"self._loss_term_weight['{k}'] * {sanitize_key(k)}" for k in out_keys]
            sum_expr = " + ".join(weighted_terms)
            lines.append("    if total_loss is None:")
            lines.append(f"        total_loss = {sum_expr}")
            lines.append("    else:")
            lines.append(f"        total_loss = total_loss + {sum_expr}")
    lines.append("    loss_log['loss'] = total_loss")
    lines.append("    return loss_log")
    return "\n".join(lines)


def generate_activation_fn(activation_records: list) -> str:
    """Return source for ``_generated_activation(self, pred) -> dict``.

    For each activation node the inputs are the ``pred.*`` entries of its
    ``node.in_key`` (the logits).  The module calls ``act.activation(*pred_logits)`` and
    stores the returned dict in the prediction workspace under the activation-node name.
    This mirrors ``_generated_forward`` / ``_generated_losses`` (teia:core/module/codegen.md).
    """
    lines = ["def _generated_activation(self, pred):"]
    lines.append("    activated = {}")
    for rec in activation_records:
        node_attr = f"self.{sanitize_key(rec.name)}"
        pred_keys = [k for k in rec.in_key if k.startswith("pred.")]
        args = ", ".join(f"pred.{k[len('pred.'):]}" for k in pred_keys)
        lines.append(f"    activated['{rec.name}'] = {node_attr}.activation({args})")
    lines.append("    return activated")
    return "\n".join(lines)


def exec_and_bind(fn_str: str, instance, attr_name: str) -> None:
    """Execute fn_str and bind the resulting function to instance as attr_name."""
    local_ns: dict = {}
    exec(fn_str, {}, local_ns)
    fn_name = fn_str.split("(")[0].replace("def ", "").strip()
    fn = local_ns[fn_name]
    setattr(instance, attr_name, types.MethodType(fn, instance))
