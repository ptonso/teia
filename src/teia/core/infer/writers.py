from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from teia.core.utils import ensure_dir


def write_predictions_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target
