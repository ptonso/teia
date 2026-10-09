from __future__ import annotations

import argparse
import importlib
import pkgutil
from typing import Sequence

import teia.core.commands as commands_pkg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="teia", description="Teia runtime CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_module in _iter_command_modules():
        command_module.register_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_teia_handler", None)
    if handler is None:
        parser.error(f"No handler registered for command: {args.command}")
    return int(handler(args) or 0)


def _iter_command_modules() -> list[object]:
    modules: list[tuple[str, object]] = []
    for module_info in pkgutil.iter_modules(commands_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module_name = f"{commands_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)
        command_name = getattr(module, "COMMAND_NAME", module_info.name.replace("_", "-"))
        modules.append((str(command_name), module))
    modules.sort(key=lambda item: item[0])
    return [module for _, module in modules]

