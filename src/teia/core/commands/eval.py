"""``teia eval``: run the eval graph offline against one or more finished runs.

Never constructs a model, datamodule, or trainer. One run rebuilds its own ``eval/`` from its
capture store; several runs drive cross-run ``Comparison`` nodes instead.

See packages/teia/specs/core/eval.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from teia.core.eval.compare_runner import run_compare
from teia.core.eval.runner import RunEvalSource, run_eval, split_evalmodule
from teia.core.utils import ensure_dir, teia_log, read_yaml

COMMAND_NAME = "eval"
HELP = "Re-run the eval graph offline against one or more finished runs."

#: Same-package relative path (teia/core/commands/eval.py -> teia/core/conf/compare/
#: default.yaml). Robust regardless of install layout, since conf/ always ships alongside
#: commands/ within this distribution. A cross-distribution teia path would instead need
#: real package-data resolution, which this command avoids.
_DEFAULT_COMPARE_BUNDLE = Path(__file__).resolve().parents[1] / "conf" / "compare" / "default.yaml"


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        dest="run_dirs",
        help="Finished run directory. Repeatable; several drive cross-run comparison.",
    )
    parser.add_argument("--split", default="test", help="Capture split to evaluate (default: test).")
    parser.add_argument("--output-dir", help="Where to write output. Defaults to <run-dir>/eval/<route> (single run) or ./compare (several).")
    parser.add_argument(
        "--compare",
        dest="compare_path",
        help="Path to a YAML file with a top-level `compare:` node map. Several --run-dir with "
        "no --compare uses the zero-config default from core/conf/compare/default.yaml "
        "(metric_table, curve_overlay, factor_diff, seed_aggregate). Copy it and add "
        "objective-specific nodes (pareto, a single-metric factor_diff or curve_overlay) for a "
        "richer comparison.",
    )
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    run_dirs = [Path(item).resolve() for item in args.run_dirs]
    if len(run_dirs) == 1:
        return _run_single(run_dirs[0], split=args.split, output_dir=args.output_dir)
    return _run_compare(run_dirs, split=args.split, output_dir=args.output_dir, compare_path=args.compare_path)


def _run_single(run_dir: Path, *, split: str, output_dir: str | None) -> int:
    source = RunEvalSource(run_dir, split=split)
    _, graph = split_evalmodule(source.config().get("evalmodule"))
    if not graph:
        teia_log("teia eval: no evalmodule nodes in this run's composed config", run_dir=str(run_dir))
        return 1
    out_dir = ensure_dir(Path(output_dir).resolve() if output_dir else run_dir / "eval")
    written = run_eval(run_dir, graph=graph, split=split, out_dir=out_dir)
    paths = [path for node_paths in written.values() for path in node_paths]
    for path in paths:
        print(path)
    teia_log("teia eval complete", run_dir=str(run_dir), outputs=len(paths))
    return 0


def _run_compare(run_dirs: list[Path], *, split: str, output_dir: str | None, compare_path: str | None) -> int:
    bundle_path = Path(compare_path) if compare_path else _DEFAULT_COMPARE_BUNDLE
    compare = read_yaml(bundle_path).get("compare") or {}
    if not compare:
        teia_log("teia eval: empty compare graph, nothing to run", compare_path=compare_path)
        return 1
    out_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / "compare"
    written = run_compare(run_dirs, compare=compare, split=split, out_dir=out_dir)
    for path in written:
        print(path)
    teia_log("teia eval compare complete", runs=len(run_dirs), outputs=len(written))
    return 0
