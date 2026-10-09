from __future__ import annotations

import warnings
from typing import Any

from teia.core.instantiate import instantiate_many


def run_hook_group(config: dict[str, Any], group: str, context: dict[str, Any]) -> None:
    hooks_cfg = config.get("hooks") or {}
    for hook in instantiate_many(hooks_cfg.get(group, [])):
        if hasattr(hook, "run"):
            hook.run(context)
        elif callable(hook):
            hook(context)
        else:
            warnings.warn(
                f"Hook {hook!r} in group {group!r} is neither callable nor has a run() method; skipping.",
                stacklevel=2,
            )

