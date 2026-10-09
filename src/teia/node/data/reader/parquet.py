"""Parquet row reader for tabular data graphs.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from teia.core.deps import require_dependency
from teia.node.data.reader.rows import FrameRows

if TYPE_CHECKING:
    import pandas as pd


class ParquetRows(FrameRows):
    """``FrameRows`` over a ``.parquet``/``.pq`` source, via ``pandas`` + ``pyarrow``."""

    default_extensions = (".parquet", ".pq")

    def read_source(self, path: Path) -> "pd.DataFrame":
        require_dependency("pyarrow", "ParquetRows")
        import pandas as pd

        return pd.read_parquet(path, **self.reader_opts)

    def write_source(self, frame: "pd.DataFrame", path: Path) -> None:
        frame.to_parquet(path, index=False)
