"""Make project-local Python importable by ``_target_`` resolution.

Hydra overlays add a project's ``conf/`` to the config searchpath, but the project
directory is never on ``sys.path`` — so a ``_target_`` pointing at project code (e.g.
``my_project.io.MyReader``) cannot import. This loader puts the project dir on ``sys.path``
once, before any datamodule/module/callback instantiation.

It is a no-op for projects whose targets all live under ``teia.*``. Side-effecting
registration (e.g. a registry call) that must run before build belongs in a
``pre_build`` hook, which now resolves project modules thanks to this loader.
"""

from __future__ import annotations

import sys
from pathlib import Path


def load_project_python(project_dir: Path | str | None) -> None:
    """Insert ``project_dir`` at the front of ``sys.path`` (idempotent)."""
    if project_dir is None:
        return
    path = str(Path(project_dir).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
