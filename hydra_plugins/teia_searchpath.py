"""Hydra search-path plugin that appends teia's ``conf/`` to the config search path.

Hydra auto-discovers any plugin under the ``hydra_plugins`` namespace package on import, so
merely installing ``teia`` registers its seed pipelines, ``node/net/*`` options and ``eval*`` groups
as if they were a project overlay, with no reference to the ``teia`` distribution in ``teia``'s
own code. ``hydra_plugins/`` must not contain an ``__init__.py``: it is a namespace package.
"""

from pathlib import Path

import teia
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class TeiaSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        home = Path(__file__).resolve().parents[1]
        for portion in map(Path, teia.__path__):
            if portion.is_relative_to(home) and (portion / "conf").is_dir():
                search_path.append(provider="teia", path=f"file://{portion / 'conf'}")
