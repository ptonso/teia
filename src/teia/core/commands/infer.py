from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.commands._runtime import add_snapshot_runtime_args, dump_cfg
from teia.core.runtime.composer import compose_run_config_tree
from teia.core.runtime.engine import infer_project


COMMAND_NAME = "infer"
HELP = "Run the infer stage."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    add_snapshot_runtime_args(parser)
    parser.add_argument(
        "--src",
        help="New input data (folder or single file) to run inference over. Read through the "
        "trained IO; the model's schema/preprocess-state stay anchored to the run.",
    )
    parser.add_argument(
        "--dst",
        help="Destination directory for the native prediction mirror + predictions.jsonl "
        "(alias of --output-dir; --dst wins if both are given).",
    )
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.cfg:
        composed, hydra_payload, *_ = compose_run_config_tree(
            run_dir=Path(args.run_dir).resolve(), overrides=list(args.overrides)
        )
        return dump_cfg(args.cfg, composed, hydra_payload)

    output_dir = args.dst or args.output_dir
    infer_project(
        run_dir=Path(args.run_dir).resolve(),
        overrides=list(args.overrides),
        checkpoint=args.checkpoint,
        output_dir=Path(output_dir).resolve() if output_dir else None,
        src=args.src,
    )
    return 0
