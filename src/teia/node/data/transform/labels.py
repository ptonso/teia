"""``EncodeLabels`` — stateful class-string → index map that publishes ``num_classes``.

The parse-alone replacement for ``PredictionDataset.classes``: it fits a vocabulary over the label stream
(all splits — dataset metadata, not a leaking statistic), maps each item's class string to an ``int64``
index, and publishes ``num_classes``/``num_labels``/``class_names`` as runtime dims the module head reads.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from teia.base.data import Transform


class EncodeLabels(Transform):
    """Stateful class-string → index encoder that publishes ``num_classes``."""
    stateful = True
    fit_all_splits = True

    def __init__(self, *, unlabeled: int = -1) -> None:
        self.unlabeled = unlabeled
        self._classes: list[str] = []
        self._index: dict[str, int] = {}

    def fit(self, train_stream: Iterable[Any]) -> None:
        for name in train_stream:
            if name is None or name in self._index:
                continue
            self._index[name] = len(self._classes)
            self._classes.append(name)

    def __call__(self, name: str | None) -> torch.Tensor:
        idx = self.unlabeled if name is None else self._index.get(name, self.unlabeled)
        return torch.tensor(idx, dtype=torch.int64)

    def runtime_dims(self) -> dict[str, Any]:
        n = len(self._classes)
        return {"num_classes": n, "num_labels": n, "class_names": list(self._classes)}

    def state(self) -> Any:
        return list(self._classes)

    def load_state(self, state: Any) -> None:
        self._classes = list(state)
        self._index = {name: i for i, name in enumerate(self._classes)}


class NameListEncode(Transform):
    """Stateful per-annotation class encoder: ``item.poly_labels (list[str]) → item.box_cls[N]`` int64.

    Fits a sorted vocabulary over every split's label lists (dataset metadata) and publishes
    ``num_classes`` — the labelme/obb analog of the YOLO ``data.yaml`` class list.
    """

    stateful = True
    fit_all_splits = True

    def __init__(self) -> None:
        self._classes: list[str] = []
        self._index: dict[str, int] = {}

    def fit(self, train_stream: Iterable[Any]) -> None:
        vocab: set[str] = set()
        for names in train_stream:
            vocab.update(n for n in (names or []) if n)
        self._classes = sorted(vocab)
        self._index = {name: i for i, name in enumerate(self._classes)}

    def __call__(self, names: list[str] | None) -> torch.Tensor:
        ids = [self._index.get(n, -1) for n in (names or [])]
        return torch.tensor(ids, dtype=torch.int64) if ids else torch.zeros((0,), dtype=torch.int64)

    def runtime_dims(self) -> dict[str, Any]:
        n = len(self._classes)
        return {"num_classes": n, "num_labels": n, "class_names": list(self._classes)}

    def state(self) -> Any:
        return list(self._classes)

    def load_state(self, state: Any) -> None:
        self._classes = list(state)
        self._index = {name: i for i, name in enumerate(self._classes)}


class MultiHotLabels(Transform):
    """Stateful multi-label encoder: ``item.label_names (list[str]) → item.cls`` multi-hot ``[num_classes]``.

    Fits a sorted vocabulary over every split's label lists (dataset metadata) and publishes
    ``num_classes``. Multi-label class-dir ingest.
    """

    stateful = True
    fit_all_splits = True

    def __init__(self) -> None:
        self._classes: list[str] = []
        self._index: dict[str, int] = {}

    def fit(self, train_stream: Iterable[Any]) -> None:
        vocab: set[str] = set()
        for names in train_stream:
            vocab.update(names or [])
        self._classes = sorted(vocab)
        self._index = {name: i for i, name in enumerate(self._classes)}

    def __call__(self, names: list[str] | None) -> torch.Tensor:
        target = torch.zeros((len(self._classes),), dtype=torch.float32)
        for name in names or []:
            idx = self._index.get(name)
            if idx is not None:
                target[idx] = 1.0
        return target

    def runtime_dims(self) -> dict[str, Any]:
        n = len(self._classes)
        return {"num_classes": n, "num_labels": n, "class_names": list(self._classes)}

    def state(self) -> Any:
        return list(self._classes)

    def load_state(self, state: Any) -> None:
        self._classes = list(state)
        self._index = {name: i for i, name in enumerate(self._classes)}
