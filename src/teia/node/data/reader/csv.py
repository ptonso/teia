"""CSV row reader for tabular data graphs.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from teia.node.data.reader.rows import FrameRows

if TYPE_CHECKING:
    import pandas as pd


class CsvRows(FrameRows):
    """``FrameRows`` over a ``.csv`` source, via ``pandas.read_csv``/``to_csv``."""

    default_extensions = (".csv",)

    def read_source(self, path: Path) -> "pd.DataFrame":
        import pandas as pd

        return pd.read_csv(path, **self.reader_opts)

    def write_source(self, frame: "pd.DataFrame", path: Path) -> None:
        frame.to_csv(path, index=False)
