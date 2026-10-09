"""``teia.core.eval``: the eval-graph resolver and runner.

See packages/teia/specs/core/eval.md.
"""

from teia.core.eval.graph import EvalNodeRecord, build_eval_graph, discover_node_entries, parse_eval_node
from teia.core.eval.runner import RunEvalSource, run_eval, run_eval_graph, run_offline_eval

__all__ = [
    "EvalNodeRecord",
    "parse_eval_node",
    "build_eval_graph",
    "discover_node_entries",
    "RunEvalSource",
    "run_eval_graph",
    "run_eval",
    "run_offline_eval",
]
