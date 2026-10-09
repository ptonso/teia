import pytest

from teia.core.utils import get_dot_path


def test_numeric_parts_index_lists() -> None:
    config = {"netmodule": {"lr_scheduler": {"scheduler": {"schedulers": [{"total_iters": 100}, {"total_iters": 1000}]}}}}
    assert get_dot_path(config, "netmodule.lr_scheduler.scheduler.schedulers.1.total_iters") == 1000


@pytest.mark.parametrize("path", ["netmodule.missing", "netmodule.items.2", "netmodule.items.x"])
def test_missing_paths_raise(path: str) -> None:
    with pytest.raises(KeyError):
        get_dot_path({"netmodule": {"items": [1, 2]}}, path)
