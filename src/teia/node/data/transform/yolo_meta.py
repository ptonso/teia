"""``YoloMeta`` — publish ``num_classes``/``class_names`` for a YOLO dataset.

Reads the dataset ``data.yaml`` (``nc``/``names``) if present, else scans the label stream for the max
class index. A plan-stage node (fits over all label paths); the ``num_classes`` it publishes is what the
detection head's ``out_key`` token resolves to.

Source: common knowledge

Description:
  See the module summary above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from teia.base.data import Transform

from teia.node.data._paths import resolve_root


class YoloMeta(Transform):
    """Plan-stage node publishing ``num_classes``/``kpt_shape`` from ``data.yaml`` or a label scan."""
    stateful = True
    fit_all_splits = True

    def __init__(self, *, root: str | None = None) -> None:
        self.root = root
        self.data_root: str | None = None
        self.project_dir: str | None = None
        self.image_size: Any = None
        self._num_classes = 0
        self._class_names: list[str] = []
        self._kpt_shape: tuple[int, int] | None = None

    def _data_yaml(self) -> dict[str, Any] | None:
        root = resolve_root(self.root, data_root=self.data_root, project_dir=self.project_dir)
        for name in ("data.yaml", "data.yml", "dataset.yaml"):
            path = root / name
            if path.is_file():
                import yaml

                return yaml.safe_load(path.read_text()) or {}
        return None

    def fit(self, train_stream: Iterable[Any]) -> None:
        meta = self._data_yaml()
        if meta is not None:
            shape = meta.get("kpt_shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                self._kpt_shape = (int(shape[0]), int(shape[1]))
        if meta is not None and (meta.get("names") or meta.get("nc")):
            names = meta.get("names")
            if isinstance(names, dict):
                names = [names[key] for key in sorted(names)]
            self._class_names = [str(n) for n in (names or [])]
            self._num_classes = int(meta.get("nc") or len(self._class_names))
            if not self._class_names:
                self._class_names = [str(i) for i in range(self._num_classes)]
            return
        max_cls = -1
        for label_path in train_stream:
            if label_path and Path(label_path).is_file():
                for line in Path(label_path).read_text().splitlines():
                    parts = line.split()
                    if parts:
                        max_cls = max(max_cls, int(float(parts[0])))
        self._num_classes = max_cls + 1
        self._class_names = [str(i) for i in range(self._num_classes)]

    def __call__(self, *inputs: Any) -> Any:  # plan-only node; never runs at item stage
        return None

    def runtime_dims(self) -> dict[str, Any]:
        dims: dict[str, Any] = {
            "num_classes": self._num_classes,
            "num_labels": self._num_classes,
            "class_names": list(self._class_names),
        }
        if self._kpt_shape is not None:
            dims["kpt_shape"] = self._kpt_shape
            dims["num_keypoints"] = self._kpt_shape[0]
        return dims

    def state(self) -> Any:
        return {"num_classes": self._num_classes, "class_names": list(self._class_names), "kpt_shape": self._kpt_shape}

    def load_state(self, state: Any) -> None:
        self._num_classes = int(state["num_classes"])
        self._class_names = list(state["class_names"])
        self._kpt_shape = tuple(state["kpt_shape"]) if state.get("kpt_shape") else None
