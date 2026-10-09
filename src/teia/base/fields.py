"""Atomic, structure-typed batch field vocabulary + composed-Batch factory.

A collate node declares the atomic fields it emits; a datamodule's ``batch_type()`` composes those
into one flat ``typing.NamedTuple`` via ``compose_batch``. See teia:base/batch.md.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, NamedTuple

from torch import Tensor

#: Structural primitive a field's signal lives on (``None`` = target / bookkeeping).
Structure = str  # {"grid", "sequence", "set", "graph"} | None


class FieldSpec(NamedTuple):
    name: str
    annotation: type  # NamedTuple field annotation (``Tensor`` or ``list``)
    structure: Structure | None
    optional: bool = False


def _f(
    name: str, annotation: type = Tensor, structure: Structure | None = None, *, optional: bool = False
) -> FieldSpec:
    return FieldSpec(name=name, annotation=annotation, structure=structure, optional=optional)


#: Authoritative atomic field inventory (see teia:base/batch.md).
FIELD_VOCAB: dict[str, FieldSpec] = {
    "image": _f("image", Tensor, "grid"),
    "text": _f("text", Tensor, "sequence"),
    "audio": _f("audio", Tensor, "grid"),
    "waveform": _f("waveform", Tensor, "sequence"),
    "tokens": _f("tokens", Tensor, "sequence"),
    "lengths": _f("lengths", Tensor),
    "numerical": _f("numerical", Tensor, "set"),
    "categorical": _f("categorical", Tensor, "set"),
    "count": _f("count", Tensor, "set"),
    "temporal": _f("temporal", Tensor, "sequence"),
    "boolean": _f("boolean", Tensor, "set"),
    "static": _f("static", Tensor, "set"),
    "numerical_mask": _f("numerical_mask", Tensor),
    "temporal_mask": _f("temporal_mask", Tensor),
    "cls": _f("cls", Tensor),
    "target": _f("target", Tensor),
    "time": _f("time", Tensor),
    "event": _f("event", Tensor),
    "labels": _f("labels", Tensor),
    "masks": _f("masks", Tensor, "grid"),
    "sem_masks": _f("sem_masks", Tensor, "grid"),
    "keypoints": _f("keypoints", Tensor),
    "bboxes": _f("bboxes", Tensor),
    "batch_idx": _f("batch_idx", Tensor),
    "ori_shape": _f("ori_shape", list),
    "path": _f("path", list),
    "polys": _f("polys", list),
    "ratio_pad": _f("ratio_pad", list),
    "action": _f("action", Tensor, optional=True),
    "reward": _f("reward", Tensor, optional=True),
    "terminated": _f("terminated", Tensor, optional=True),
    "truncated": _f("truncated", Tensor, optional=True),
    "log_prob": _f("log_prob", Tensor, optional=True),
    "value": _f("value", Tensor, optional=True),
    "advantage": _f("advantage", Tensor, optional=True),
    "return_": _f("return_", Tensor, optional=True),
    "episode_start": _f("episode_start", Tensor, optional=True),
    "mask": _f("mask", Tensor, optional=True),
    "episode_id": _f("episode_id", Tensor, optional=True),
    "weight": _f("weight", Tensor, optional=True),
    "index": _f("index", Tensor, optional=True),
    "n_step_discount": _f("n_step_discount", Tensor, optional=True),
    "obs_state": _f("obs_state", Tensor, "set", optional=True),
    "obs_pixels": _f("obs_pixels", Tensor, "grid", optional=True),
    "next_obs_state": _f("next_obs_state", Tensor, "set", optional=True),
    "next_obs_pixels": _f("next_obs_pixels", Tensor, "grid", optional=True),
}


@lru_cache(maxsize=None)
def _compose(names: tuple[str, ...]) -> type:
    """Build the NamedTuple, module-registered under a unique name so instances survive pickling
    across DataLoader workers.

    ``typing.NamedTuple``'s generated ``__new__`` accepts defaults for a trailing run of fields only,
    so this gives every optional field in ``names`` a ``None`` default iff all optional fields in
    ``FIELD_VOCAB`` sort last (true today: non-optional fields are declared first, optional ones
    appended after). A ``names`` list with an optional field ahead of a required one would need
    ``typing.NamedTuple``'s functional form invoked directly with defaults threaded in, not this
    ``lru_cache``'d path.
    """
    fields = []
    specs = []
    for name in names:
        spec = FIELD_VOCAB.get(name)
        if spec is None:
            raise KeyError(f"Unknown atomic batch field {name!r}; register it in teia.base.fields.FIELD_VOCAB.")
        fields.append((name, spec.annotation))
        specs.append(spec)
    type_name = "Batch__" + "_".join(names)
    cls = NamedTuple(type_name, fields)  # type: ignore[misc]
    cls.__qualname__ = type_name
    n_optional = sum(1 for s in specs if s.optional)
    if n_optional:
        cls.__new__.__defaults__ = (None,) * n_optional
        cls._field_defaults = {s.name: None for s in specs if s.optional}
    globals()[type_name] = cls
    return cls


def compose_batch(names: Iterable[str]) -> type:
    """Build (cached) the flat ``Batch`` NamedTuple over the given atomic fields, order preserved."""
    return _compose(tuple(names))


def register_field(name: str, *, structure: Structure | None = None, optional: bool = True) -> None:
    """Escape hatch for names not known in advance (exotic multi-modal dict-obs backends emitting
    ``obs_<key>`` for an arbitrary key). Idempotent — re-registering the same name with the same spec
    is a no-op; re-registering with a different spec is a hard error (catches accidental collisions)."""
    existing = FIELD_VOCAB.get(name)
    spec = _f(name, structure=structure, optional=optional)
    if existing is not None and existing != spec:
        raise ValueError(f"register_field({name!r}, ...) conflicts with existing FIELD_VOCAB entry {existing!r}.")
    FIELD_VOCAB[name] = spec
