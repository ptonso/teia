from __future__ import annotations

import argparse
from typing import Any


def _add_cfg_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--cfg",
        choices=("job", "hydra", "all"),
        default=None,
        help="Print the composed config (job|hydra|all) as YAML and exit.",
    )


def dump_cfg(cfg: str, composed: dict[str, Any], hydra_payload: dict[str, Any] | None) -> int:
    import yaml

    payload = {"job": composed, "hydra": hydra_payload, "all": {"job": composed, "hydra": hydra_payload}}[cfg]
    print(yaml.safe_dump(payload, sort_keys=False).rstrip())
    return 0


def add_project_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", default=".", help="Project root containing conf/.")
    parser.add_argument("--config-dir", help="Explicit config directory. Defaults to <project>/conf.")
    parser.add_argument("--run-name", help="Optional run-name override.")
    _add_cfg_arg(parser)
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides.")


def add_train_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", help="Project root containing conf/. Defaults to the current directory.")
    parser.add_argument("--config-dir", help="Explicit config directory. Defaults to <project>/conf.")
    parser.add_argument("--run-name", help="Optional run-name override.")
    parser.add_argument("--from-run-dir", help="Existing run directory whose saved config snapshot seeds a continued train phase.")
    parser.add_argument(
        "--from-run-weights",
        choices=("best", "last"),
        default="best",
        help="Checkpoint policy used with --from-run-dir when --checkpoint is not provided.",
    )
    parser.add_argument("--checkpoint", help="Optional checkpoint path override.")
    _add_cfg_arg(parser)
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides.")


def add_snapshot_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing config/conf/.")
    parser.add_argument("--checkpoint", help="Optional checkpoint path override.")
    parser.add_argument("--output-dir", help="Optional explicit stage output directory.")
    _add_cfg_arg(parser)
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides applied to the saved run snapshot.")
