from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.runtime.sweep import run_sweep


COMMAND_NAME = "sweep"
HELP = "Search hyperparameters: every sweep expression in the overrides becomes a dimension."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    parser.add_argument("--project-dir", help="Project root containing conf/. Defaults to the current directory.")
    parser.add_argument("--config-dir", help="Explicit config directory. Defaults to <project>/conf.")
    parser.add_argument("--run-name", help="Optional sweep-name override.")
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides; sweep expressions (a=1,2 / interval(...)) become dimensions.")
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    _, best = run_sweep(
        project_dir=Path(args.project_dir) if args.project_dir else None,
        config_dir=Path(args.config_dir).resolve() if args.config_dir else None,
        overrides=list(args.overrides),
        run_name=args.run_name,
    )
    print("\n".join(best))
    return 0
