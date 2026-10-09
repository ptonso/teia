"""Generic ``.teia-cache`` index reuse.

The signature/load/write helpers let a reader skip re-scanning an unchanged dataset: a reader mixes them in and
stores its index under ``.teia-cache``. See teia:core/datamodule.md (cache controls).
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any


def signature_entry(path: Path) -> tuple:
    stat = path.stat()
    if path.is_dir():
        return ("dir", path.name, int(stat.st_mtime_ns), sum(1 for _ in os.scandir(path)))
    if path.is_file():
        return ("file", path.name, int(stat.st_mtime_ns), int(stat.st_size))
    raise FileNotFoundError(path)


class IndexCacheMixin:
    """Sign a reader's index by the mtime/size of its roots and persist it under ``.teia-cache``.

    A reader supplies ``cache_roots()`` (paths that sign the cache), ``cache_key_parts()`` (identity
    strings distinguishing this reader's index), and ``cache_subdir``/``project_dir``. It then wraps a
    scan with ``load_index()`` / ``write_index(index)``. ``project_dir is None`` disables caching.
    """

    cache_subdir: str = "index"
    project_dir: str | None = None

    def cache_roots(self) -> list[Path]:
        raise NotImplementedError

    def cache_key_parts(self) -> tuple:
        raise NotImplementedError

    def _cache_path(self) -> Path | None:
        if self.project_dir is None:
            return None
        key = hashlib.sha1("|".join(str(p) for p in self.cache_key_parts()).encode()).hexdigest()
        return Path(self.project_dir) / ".teia-cache" / self.cache_subdir / f"{key}.pkl"

    def _signature(self) -> tuple:
        return tuple(sorted(signature_entry(path) for path in self.cache_roots()))

    def load_index(self) -> Any | None:
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            payload = pickle.loads(path.read_bytes())
            if payload["signature"] == self._signature():
                return payload["index"]
        except Exception:
            path.unlink(missing_ok=True)
        return None

    def write_index(self, index: Any) -> None:
        path = self._cache_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"signature": self._signature(), "index": index}))
