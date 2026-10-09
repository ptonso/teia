from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.commands._runtime import add_train_runtime_args, dump_cfg
from teia.core.runtime.composer import compose_overlay_config, compose_run_config_tree
from teia.core.runtime.engine import train_project


COMMAND_NAME = "train"
HELP = "Run the train stage."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    add_train_runtime_args(parser)
    parser.add_argument("--skip-post-test", action="store_true", help="Skip the post-train test stage.")
    parser.add_argument("--skip-report", action="store_true", help="Skip the post-train eval graph.")
    parser.add_argument(
        "--export",
        dest="skip_export",
        action="store_false",
        default=None,
        help="Run post-train export execution.",
    )
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    from_run_dir = Path(args.from_run_dir).resolve() if args.from_run_dir else None
    if from_run_dir is not None:
        if args.project_dir is not None:
            raise ValueError("--project-dir cannot be used with --from-run-dir.")
        if args.config_dir is not None:
            raise ValueError("--config-dir cannot be used with --from-run-dir.")
        project_dir = None
        config_dir = None
    else:
        project_dir = Path(args.project_dir or ".").resolve()
        config_dir = Path(args.config_dir).resolve() if args.config_dir else None

    if args.cfg:
        if from_run_dir is not None:
            composed, hydra_payload, *_ = compose_run_config_tree(run_dir=from_run_dir, overrides=list(args.overrides))
        else:
            composed, hydra_payload = compose_overlay_config(
                project_dir=project_dir, config_dir=config_dir, overrides=list(args.overrides)
            )
        return dump_cfg(args.cfg, composed, hydra_payload)

    train_project(
        project_dir=project_dir,
        config_dir=config_dir,
        overrides=list(args.overrides),
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        from_run_dir=from_run_dir,
        from_run_weights=args.from_run_weights,
        skip_post_test=args.skip_post_test,
        skip_report=args.skip_report,
        skip_export=args.skip_export,
    )
    return 0
