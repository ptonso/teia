"""Pipeline-key string helpers (shared floor).

``sanitize_key`` (dot-syntax key → valid Python identifier) is used by engine codegen and callbacks
that resolve module aliases, so it lives in ``teia.base``. Stdlib-only.
"""

from __future__ import annotations


def sanitize_key(key: str) -> str:
    """Convert a dot-syntax key to a valid Python identifier for local vars.

    ``feat.pooled``  → ``feat_pooled``
    ``pred.logits``  → ``pred_logits``
    ``batch.image``  → ``batch_image``
    """
    return key.replace(".", "_")
