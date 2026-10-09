from __future__ import annotations

from typing import Any


def _missing(feature: str) -> RuntimeError:
    return RuntimeError(
        f"{feature} is still using the built-in placeholder object. "
        "Replace the default scaffold target with your project implementation."
    )


class PlaceholderDataModule:
    def __init__(self, **_: Any) -> None:
        self.config = _

    def setup(self, stage: str | None = None) -> None:
        raise _missing("Data module")


class PlaceholderReport:
    def __init__(self, **_: Any) -> None:
        self.config = _

    def run(self, context: dict[str, Any]):
        return []
