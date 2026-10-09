"""SQL row reader for tabular data graphs.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path

from teia.core.deps import require_dependency
from teia.base.schema import SchemaError
from teia.node.data._frame import split_frame
from teia.node.data.reader.rows import FrameRows


class SqlRows(FrameRows):
    """``FrameRows`` over a SQL query result, via ``sqlalchemy`` + ``pandas.read_sql``."""

    def _engine(self):
        require_dependency("sqlalchemy", "SqlRows")
        from sqlalchemy import create_engine

        uri = self.reader_opts.get("uri")
        if not uri:
            raise SchemaError("SqlRows requires `uri`.")
        return create_engine(str(uri))

    def _load_frame(self, split: str):
        import pandas as pd

        queries = self.reader_opts.get("queries") or {}
        query = queries.get(split) or self.reader_opts.get("query")
        if not query:
            raise SchemaError(f"SqlRows has no query for split {split!r}.")
        frame = pd.read_sql(str(query), self._engine())
        return frame.reset_index(drop=True) if split in queries else split_frame(frame, split, self.split_controls)

    def read_source(self, path: Path):
        raise NotImplementedError("SqlRows reads from SQL queries, not file paths.")
