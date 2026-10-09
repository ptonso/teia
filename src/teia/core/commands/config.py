from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.config.explain import explain_overrides
from teia.core.config.layers import resolve_conf_layout
from teia.core.config.lint import package_address, run_lint
from teia.core.config.scaffold import write_overlay_scaffold, write_resolved_task_scaffold
from teia.core.runtime.composer import compose_overlay_config
from teia.core.utils import get_dot_path

COMMAND_NAME = "config"
HELP = "Inspect and scaffold Teia configuration."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    config_subparsers = parser.add_subparsers(dest="config_command", required=True)

    init_parser = config_subparsers.add_parser("init", help="Write project node-leaf stubs under conf/node/{net,data}.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing override stubs.")
    init_parser.add_argument(
        "--from",
        dest="from_task",
        help="Materialize task=<name>'s three modules, fully resolved, under conf/<graph>module/mine/.",
    )
    init_parser.set_defaults(_teia_handler=_run_init)

    list_parser = config_subparsers.add_parser("list", help="List the config layers Teia composes from.")
    list_parser.set_defaults(_teia_handler=_run_list)

    show_parser = config_subparsers.add_parser("show", help="Show a config group option.")
    show_parser.add_argument("selector", help="Config group selector, e.g. node/net/encoder=mlp.")
    show_parser.add_argument("--resolved", action="store_true", help="Show the composed subtree at the option's `# @package` address (the whole config for `_global_` bundles).")
    show_parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides used with --resolved.")
    show_parser.set_defaults(_teia_handler=_run_show)

    tree_parser = config_subparsers.add_parser("tree", help="Print the config-group directory tree across all layers.")
    tree_parser.add_argument("--root", help="Scope to a group path, e.g. node/net/encoder.")
    tree_parser.add_argument("--level", type=int, default=None, help="Max depth to descend (default: unlimited).")
    tree_parser.add_argument("--node-only", action="store_true", help="Show only groups (folders), hiding yaml options.")
    tree_parser.set_defaults(_teia_handler=_run_tree)

    explain_parser = config_subparsers.add_parser(
        "explain", help="Attribute each changed config group's final option to the override (or task preset) that set it."
    )
    explain_parser.add_argument("overrides", nargs="+", help="Overrides to explain, e.g. task=vision-cls netmodule=vision-cls/mlp.")
    explain_parser.set_defaults(_teia_handler=_run_explain)

    lint_parser = config_subparsers.add_parser(
        "lint", help="Check node leaves, modules and task presets across the real conf tree."
    )
    lint_parser.set_defaults(_teia_handler=_run_lint)

    parser.set_defaults(_teia_handler=lambda args: parser.error("Specify a config subcommand: init, list, show, tree, explain, lint."))
    return parser


def _run_init(args: argparse.Namespace) -> int:
    root = Path(".").resolve()
    if args.from_task:
        group, _, name = args.from_task.partition("=")
        if group != "task" or not name:
            raise ValueError(f"--from expects task=<name>, got {args.from_task!r}.")
        written = write_resolved_task_scaffold(root=root, task=name, force=args.force)
    else:
        written = write_overlay_scaffold(root=root, force=args.force)
    if not written:
        print("Nothing written (stubs already exist; use --force to overwrite).")
    for path in written:
        print(f"wrote {path}")
    return 0


def _run_explain(args: argparse.Namespace) -> int:
    rows = explain_overrides(Path(".").resolve(), list(args.overrides))
    if not rows:
        print("No config group changed relative to the zero-override baseline.")
        return 0
    width = max(len(group) for group, _, _ in rows)
    for group, option, source in rows:
        print(f"{group.replace('/', '.'):<{width}} {option:<12} <- {source}")
    return 0


def _run_lint(args: argparse.Namespace) -> int:
    findings = run_lint(Path(".").resolve())
    for finding in findings:
        print(finding)
    if findings:
        print(f"\n{len(findings)} lint finding(s).")
        return 1
    print("Lint clean.")
    return 0


def _run_list(args: argparse.Namespace) -> int:
    layout = resolve_conf_layout(Path(".").resolve())
    print("Config layers:")
    for layer in layout.ordered:
        print(f"  {layer.label:<10} {layer.uri}")
    if layout.project is None:
        print("  project    (none — run `teia config init` to add overrides)")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    if "=" not in args.selector:
        raise ValueError("Selector must be of the form group=option, e.g. node/net/encoder=mlp.")
    group, option = args.selector.split("=", 1)
    group_path = group.replace(".", "/")

    layout = resolve_conf_layout(Path(".").resolve())
    for layer in layout.ordered:
        candidate = layer.path / group_path / f"{option}.yaml"
        if candidate.exists():
            break
    else:
        raise FileNotFoundError(f"No config file found for {args.selector!r} in any layer.")

    if args.resolved:
        import yaml

        if group_path.startswith("node/"):  # a leaf is only placed by a mounting module: resolved is itself
            print(yaml.safe_dump(yaml.safe_load(candidate.read_text(encoding="utf-8")), sort_keys=False).rstrip())
            return 0
        composed, _ = compose_overlay_config(
            project_dir=Path(".").resolve(),
            overrides=[args.selector, *args.overrides],
        )
        address = package_address(candidate) or group_path.replace("/", ".")
        subtree = composed if address == "_global_" else get_dot_path(composed, address)
        print(yaml.safe_dump(subtree, sort_keys=False).rstrip())
        return 0

    print(f"# layer: {layer.label}")
    print(f"# source: {candidate}")
    print(candidate.read_text(encoding="utf-8").rstrip())
    return 0


def _run_tree(args: argparse.Namespace) -> int:
    if args.level is not None and args.level < 1:
        raise ValueError("--level must be >= 1.")

    layout = resolve_conf_layout(Path(".").resolve())
    rel = args.root.replace(".", "/").strip("/") if args.root else ""
    bases = [(layer.label, layer.path / rel if rel else layer.path) for layer in layout.ordered]
    bases = [(label, base) for label, base in bases if base.is_dir()]
    if not bases:
        searched = ", ".join(layer.label for layer in layout.ordered)
        raise FileNotFoundError(f"No config group {args.root!r} found in any layer (searched: {searched}).")

    lines = [f"{args.root or 'conf'}  {_layer_tag(label for label, _ in bases)}"]
    _render_tree(bases, lines, prefix="", depth=1, max_level=args.level, node_only=args.node_only)
    print("\n".join(lines))
    return 0


def _layer_tag(labels) -> str:
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return f"[{'+'.join(seen)}]"


def _render_tree(
    bases: list[tuple[str, Path]],
    lines: list[str],
    *,
    prefix: str,
    depth: int,
    max_level: int | None,
    node_only: bool,
) -> None:
    subdirs: dict[str, list[tuple[str, Path]]] = {}
    options: dict[str, list[str]] = {}
    for label, base in bases:
        for entry in sorted(base.iterdir(), key=lambda path: path.name):
            if entry.is_dir():
                subdirs.setdefault(entry.name, []).append((label, entry))
            elif entry.suffix in (".yaml", ".yml") and not node_only:
                options.setdefault(entry.stem, []).append(label)

    items: list[tuple[str, bool, list]] = [
        *((name, True, layered) for name, layered in subdirs.items()),
        *((name, False, layers) for name, layers in options.items()),
    ]
    for index, (name, is_dir, payload) in enumerate(items):
        last = index == len(items) - 1
        connector = "└─ " if last else "├─ "
        suffix = "/" if is_dir else ""
        tag = _layer_tag(label for label, _ in payload) if is_dir else _layer_tag(payload)
        lines.append(f"{prefix}{connector}{name}{suffix}  {tag}")
        if is_dir and (max_level is None or depth < max_level):
            extension = "   " if last else "│  "
            _render_tree(
                payload,
                lines,
                prefix=prefix + extension,
                depth=depth + 1,
                max_level=max_level,
                node_only=node_only,
            )
