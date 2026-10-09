from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.commands._runtime import add_snapshot_runtime_args, dump_cfg
from teia.core.runtime.composer import compose_run_config_tree
from teia.core.runtime.engine import export_project


COMMAND_NAME = "export"
HELP = "Run the export stage."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    add_snapshot_runtime_args(parser)
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.cfg:
        composed, hydra_payload, *_ = compose_run_config_tree(
            run_dir=Path(args.run_dir).resolve(), overrides=list(args.overrides)
        )
        return dump_cfg(args.cfg, composed, hydra_payload)

    export_project(
        run_dir=Path(args.run_dir).resolve(),
        overrides=list(args.overrides),
        checkpoint=args.checkpoint,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    return 0
