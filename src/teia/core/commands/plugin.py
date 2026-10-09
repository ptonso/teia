from __future__ import annotations

import argparse
from pathlib import Path


COMMAND_NAME = "plugin"
HELP = "Check that a project is a teia plugin (specs/plugin.md)."


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    actions = parser.add_subparsers(dest="plugin_command", required=True)
    check = actions.add_parser("check", help="report every rule a plugin folder violates")
    check.add_argument("path", nargs="?", default=".", help="plugin folder holding pyproject.toml (default: .)")
    check.add_argument("--publish", action="store_true", help="also fail on what blocks publishing")
    check.add_argument("--namespace", default=None, help="package that holds the plugin's Python (default: from the distribution name)")
    check.set_defaults(_teia_handler=run_check)
    init = actions.add_parser("init", help="create the files a plugin needs; existing files are kept")
    init.add_argument("path", nargs="?", default=".", help="project folder (default: .)")
    init.add_argument("--name", required=True, help="distribution name, lowercase; the package name replaces - with _")
    init.add_argument("--license", default="Apache-2.0", help="SPDX license identifier (default: Apache-2.0)")
    init.set_defaults(_teia_handler=run_init)
    return parser


def run_init(args: argparse.Namespace) -> int:
    from teia.core.plugin import scaffold

    root = Path(args.path).resolve()
    for path, created in scaffold(root, args.name, args.license):
        print(f"{'created' if created else 'kept   '} {path.relative_to(root)}")
    print(f"Next: add a LICENSE file, pip install -e {root}, then teia plugin check {root}")
    return 0


def run_check(args: argparse.Namespace) -> int:
    from teia.core.plugin import check

    findings = check(Path(args.path), publish=args.publish, namespace=args.namespace)
    for finding in findings:
        print(finding)
    errors = sum(finding.level == "error" for finding in findings)
    warnings = len(findings) - errors
    print(f"{errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0
