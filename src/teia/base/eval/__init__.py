"""``teia.base.eval``: the eval-node ABCs.

Imported directly (``from teia.base.eval import Metric, ...``), like ``teia.base.data`` and
``teia.base.net``. Ships no concrete metric, view, comparison, or kernel.
"""

from teia.base.eval.nodes import Comparison, EvalNode, EvalSource, Metric, View

__all__ = ["EvalNode", "Metric", "View", "Comparison", "EvalSource"]
