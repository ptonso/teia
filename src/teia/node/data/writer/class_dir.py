"""``ClassDirWriter`` — materialize classification predictions as a class-folder tree.

Source: common knowledge

Description:
  The inverse of ``AudioFolderReader``/``ImageFolderReader``: each source file is copied into
  ``dst/<predicted_class>/<filename>``. Wired ``in: [capture.<route>.label, batch.path, meta.class_names]``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from teia.base.data import Writer


class ClassDirWriter(Writer):
    """``(label, source path, class names) → dst/<class_name>/<filename>`` folder tree."""

    def __call__(self, labels: Any, paths: Any, class_names: Any, *, dst: Path) -> list[Path]:
        names = [str(name) for name in class_names]
        for label, source in zip(list(labels), list(paths)):
            src = Path(str(source))
            target = Path(dst) / names[int(label)] / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        return [Path(dst)]
