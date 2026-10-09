from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.commands._runtime import add_snapshot_runtime_args, dump_cfg
from teia.core.runtime.composer import compose_run_config_tree
from teia.core.runtime.engine import test_project


COMMAND_NAME = "test"
HELP = "Run the test stage."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    add_snapshot_runtime_args(parser)
    parser.add_argument("--skip-report", action="store_true", help="Skip the post-test eval graph.")
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.cfg:
        composed, hydra_payload, *_ = compose_run_config_tree(
            run_dir=Path(args.run_dir).resolve(), overrides=list(args.overrides)
        )
        return dump_cfg(args.cfg, composed, hydra_payload)

    test_project(
        run_dir=Path(args.run_dir).resolve(),
        overrides=list(args.overrides),
        checkpoint=args.checkpoint,
        skip_report=args.skip_report,
    )
    return 0
