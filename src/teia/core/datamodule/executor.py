"""``TeiaDataModule`` — the single core ``LightningDataModule`` that executes an data graph.

Generic executor replacing the per-modality datamodules: it builds the graph's nodes, runs the
plan → item → batch lifecycle, and exposes the ``batch_type()``/``batch_meta()`` + task-metadata
protocol the module dry-run consumes. Concrete data nodes live in ``teia.node.data.*``; this class ships
none. See teia:core/datamodule.md.

Multi-reader graphs are aligned by a single ``join`` node (``how ∈ {inner,outer,left}``, ``on: key``)
at the plan stage; a lone reader needs no join. Runtime dims (``num_classes``/``kpt_shape``/…) are
published by nodes via ``runtime_dims()`` and surfaced as plain attributes so a module head's
``out_key: [num_classes]`` resolves via ``getattr``.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Any

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import torch

from teia.base.fields import FIELD_VOCAB, compose_batch, register_field
from teia.base.data import Collate, Join, Reader, Transform
from teia.core.datamodule.graph import IoNodeRecord, batch_field_names, build_graph, discover_node_entries, join_input_name
from teia.core.instantiate import instantiate

_SPLITS = ("train", "val", "test")
#: Shared datamodule fields injected onto any node that declares the attribute (defaulted to ``None``).
_SHARED_NODE_ATTRS = ("project_dir", "data_root", "image_size")
_UNSET: Any = object()


def _assign(workspace: dict[str, Any], out_key: list[str], result: Any) -> None:
    if not out_key:
        return
    if len(out_key) == 1:
        workspace[out_key[0]] = result
        return
    for key, value in zip(out_key, result):
        workspace[key] = value


def _as_tuple(value: Any) -> tuple:
    return value if isinstance(value, tuple) else (value,)


def _resolve_workers(value: int | str, *, cap: int) -> int:
    if value == "auto":
        return min(int(os.cpu_count() or 0), cap)
    return int(value)


class _ItemDataset(Dataset):
    """Runs the ``item`` subgraph per sample: seed the root (reader/join) output, then item-stage nodes.

    ``augment``-phase item nodes run for ``split == "train"`` only; eval/test/predict get the
    deterministic letterbox core (keeping the export/infer slice, which drops ``augment`` by phase,
    consistent with what training's non-augment path produces)."""

    def __init__(self, dm: TeiaDataModule, split: str) -> None:
        self._dm = dm
        self._split = split
        self._seeds = dm._seeds_for(split)

    def __len__(self) -> int:
        return len(self._seeds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dm._item_workspace(self._seeds[index], augment=self._split == "train")


class TeiaDataModule(LightningDataModule):
    """Executes an data graph.

    The graph arrives as ``**nodes`` — each data-node envelope is a named kwarg alongside the shared
    fields (as module aliases arrive as kwargs to ``TeiaNetModule``). Nodes are ``_target_``-
    instantiated so core never imports ``teia.node``; the executor injects each node's envelope.
    """

    #: TaskDataModule contract fields (see teia.base.protocols).
    eval_capture_only: bool = False

    def __init__(
        self,
        *,
        batch_size: Any = _UNSET,
        num_workers: Any = _UNSET,
        val_num_workers: int | str | None = None,
        pin_memory: bool = False,
        persistent_workers: bool | str = False,
        drop_last: bool = False,
        data_root: str | None = None,
        project_dir: str | None = None,
        image_size: Any = None,
        pixel_norm: str | None = None,
        eval_capture_only: bool = False,
        runner: Any = None,
        view: Any = None,
        **nodes: Any,
    ) -> None:
        super().__init__()
        self.view = view
        self.eval_capture_only = eval_capture_only
        self._context: dict[str, Any] = {}
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.data_root = data_root
        self.project_dir = project_dir
        self.image_size = image_size
        self.pixel_norm = pixel_norm
        self.eval_capture_only = eval_capture_only
        self.train_split, self.val_split, self.test_split = _SPLITS

        # Dataset facts published by nodes' ``runtime_dims()`` at plan time (see ``meta()``).
        self._runtime_dims: dict[str, Any] = {}
        self._planned = False

        shared = {"project_dir": project_dir, "data_root": data_root, "image_size": image_size}
        self._records: list[IoNodeRecord] = build_graph(discover_node_entries(nodes))
        self._live = any(rec.stage == "stream" for rec in self._records)
        self._nodes: dict[str, Any] = {}
        for rec in self._records:
            node = instantiate(rec.component)
            node.in_key, node.out_key = rec.in_key, rec.out_key
            node.phase, node.stage = rec.phase, rec.stage
            for attr in _SHARED_NODE_ATTRS:
                if hasattr(node, attr) and getattr(node, attr, None) is None:
                    setattr(node, attr, shared[attr])
            self._nodes[rec.name] = node

        self._readers = [r for r in self._records if r.kind == "read"]
        if not self._readers:
            raise ValueError("data graph declares no reader; at least one `read` node is required.")
        joins = [r for r in self._records if r.kind == "join"]
        if len(joins) > 1:
            raise NotImplementedError(f"TeiaDataModule supports at most one join; got {len(joins)}.")
        self._join_record: IoNodeRecord | None = joins[0] if joins else None
        if self._join_record is None and len(self._readers) > 1:
            raise ValueError("multiple readers require a `join` node to align them.")

        n_out = 0
        if self._join_record is not None:
            n_in = len(self._join_record.in_key)
            n_out = n_in * (2 if self._nodes[self._join_record.name].how == "outer" else 1)
            if len(self._join_record.out_key) != n_out:
                raise ValueError(
                    f"join '{self._join_record.name}': out_key needs {n_out} keys (raw per input"
                    f"{', then present per input' if n_out > n_in else ''}); got {self._join_record.out_key}."
                )
            self._present_fields = {
                key: f"{join_input_name(inp)}_present"
                for inp, key in zip(self._join_record.in_key, self._join_record.out_key[n_in:])
            }
        else:
            self._present_fields = {}
        self._field_names = batch_field_names(
            self._records, prefix=self._join_record is not None and self._nodes[self._join_record.name].prefix
        )
        for rec in self._records:
            if rec.kind == "collate" and self._field_names[rec.name] != rec.field:
                base = FIELD_VOCAB[rec.field]
                register_field(self._field_names[rec.name], structure=base.structure, optional=base.optional)
        for field in self._present_fields.values():
            register_field(field, optional=False)

        self._plan_reshape_records = [r for r in self._records if r.stage == "plan" and r.kind == "reshape"]
        self._writer_records = [r for r in self._records if r.kind == "write"]
        self._item_records = [r for r in self._records if r.stage == "item" and r.kind not in {"collate", "write"}]
        #: Train-only stochastic augments (phase ``augment``); rewrites, so skipping them on eval leaves
        #: the deterministic letterbox values in place for the downstream (raster/collate) consumers.
        self._augment_records = [r for r in self._item_records if r.phase == "augment"]
        self._collate_records = [r for r in self._records if r.kind == "collate"]
        self._post_transfer_records = [
            r for r in self._records if r.stage == "batch" and r.kind not in {"collate", "write"}
        ]
        self._base_seeds: dict[str, list[dict[str, Any]]] = {}
        self._seeds: dict[str, list[dict[str, Any]]] = {}
        self._predict_split = "test"

        self._regime: Any = None
        self._runner: Any = None
        if self._live:
            from teia.core.datamodule.interactive import InteractiveRegime, Runner, SyncRunner

            if num_workers is not _UNSET and _resolve_workers(num_workers, cap=16) != 0:
                raise ValueError(f"a live (stream) graph runs num_workers=0 (inline collection); got {num_workers!r}.")
            if batch_size is not _UNSET and batch_size is not None:
                raise ValueError(f"a live (stream) graph has batch_size=None (the runner yields whole minibatches); got {batch_size!r}.")
            self.batch_size = None
            self.num_workers = 0
            self.val_num_workers = 0
            self.persistent_workers = False
            self._regime = InteractiveRegime(self)
            if isinstance(runner, Runner):
                self._runner = runner
            elif runner is not None:
                self._runner = instantiate(runner)
            else:
                self._runner = SyncRunner()
        else:
            self.batch_size = 8 if batch_size is _UNSET else batch_size
            self.num_workers = _resolve_workers(4 if num_workers is _UNSET else num_workers, cap=16)
            self.val_num_workers = (
                self.num_workers if val_num_workers is None else _resolve_workers(val_num_workers, cap=8)
            )
            self.persistent_workers = (
                self.num_workers > 0 if persistent_workers == "auto" else bool(persistent_workers)
            )

    # -- context (ctx.*) workspace -------------------------------------------------
    def bind_context(self, **live: Any) -> None:
        """Bind live, externally-produced values a ``stream``-stage node's ``ctx.<name>`` in_key
        resolves to (e.g. ``bind_context(policy=module)``). Test/explicit override; Lightning-native
        resolution (``resolve_context``) covers the common case."""
        self._context.update(live)

    def resolve_context(self, name: str) -> Any:
        if name in self._context:
            return self._context[name]
        if name == "policy":
            trainer = getattr(self, "trainer", None)
            if trainer is not None and getattr(trainer, "lightning_module", None) is not None:
                return trainer.lightning_module
        raise RuntimeError(f"No ctx.{name} bound: call bind_context({name}=...) or run under Trainer.fit.")

    # -- stream (live) graph accessors ---------------------------------------------
    def _stream_reader_node(self) -> Any:
        for rec in self._readers:
            if rec.stage == "stream":
                return self._nodes[rec.name]
        raise RuntimeError("live graph has no stage='stream' reader.")

    def _stream_reshape_node(self) -> Any | None:
        for rec in self._records:
            if rec.stage == "stream" and rec.kind == "reshape":
                return self._nodes[rec.name]
        return None

    def _bind_stream_context(self) -> dict[str, Any]:
        names = {key[len("ctx.") :] for rec in self._records for key in rec.in_key if key.startswith("ctx.")}
        return {name: self.resolve_context(name) for name in names}

    def _runner_context(self, *, training: bool) -> Any:
        from teia.core.datamodule.interactive.regime import _ensure_episode_log, _policy_device
        from teia.core.datamodule.interactive.runner_base import RunnerContext

        reader = self._stream_reader_node()
        module = self.resolve_context("policy")
        return RunnerContext(
            dm=self,
            stream_reader=reader,
            stream_reshape=self._stream_reshape_node(),
            ctx=self._bind_stream_context(),
            total_env_steps=reader.total_env_steps,
            num_envs=reader.num_envs,
            seed=reader.seed,
            device=_policy_device(module),
            generator=reader.sampling_generator(),
            training=training,
            episode_log=_ensure_episode_log(module) if training else None,
        )

    @property
    def eval_episodes(self) -> int:
        return int(self._stream_reader_node().eval_episodes)

    def evaluate(self, module: Any, *, episodes: int = 10, deterministic: bool = True, seed: int | None = None) -> dict[str, list[float]]:
        return self._regime.evaluate(module, episodes=episodes, deterministic=deterministic, seed=seed)

    # -- plan stage ---------------------------------------------------------------
    def _build_streams(self, split: str) -> dict[str, dict[Any, Any]]:
        """Each reader's indexed stream ``{key → raw}`` for ``split``, keyed by the reader's out_key."""
        return {r.out_key[0]: dict(self._nodes[r.name].iter_split(split)) for r in self._readers}

    def _base_seeds_for(self, split: str) -> list[dict[str, Any]]:
        """Reader/join seeds before plan-stage reshapes alter item granularity."""
        if split in self._base_seeds:
            return self._base_seeds[split]
        streams = self._build_streams(split)
        if self._join_record is not None:
            join: Join = self._nodes[self._join_record.name]
            rows = join(*[streams[k] for k in self._join_record.in_key])
            seeds = [dict(zip(self._join_record.out_key, _as_tuple(row))) for row in rows]
        else:
            out_key = self._readers[0].out_key
            seeds = []
            for raw in streams[out_key[0]].values():
                seed: dict[str, Any] = {}
                _assign(seed, out_key, raw)
                seeds.append(seed)
        self._base_seeds[split] = seeds
        return seeds

    def _seeds_for(self, split: str) -> list[dict[str, Any]]:
        """Per-item root seeds for ``split`` after plan-stage reshapes."""
        if split in self._seeds:
            return self._seeds[split]
        seeds = [dict(seed) for seed in self._base_seeds_for(split)]
        for rec in self._plan_reshape_records:
            node = self._nodes[rec.name]
            columns = [[seed[k] for seed in seeds] for k in rec.in_key]
            rows = list(node(*columns))
            next_seeds: list[dict[str, Any]] = []
            for row in rows:
                seed = dict(row) if isinstance(row, dict) else {}
                if not isinstance(row, dict):
                    _assign(seed, rec.out_key, row)
                next_seeds.append(seed)
            seeds = next_seeds
        self._seeds[split] = seeds
        return seeds

    def ensure_task_metadata(self) -> None:
        """TaskMetadataReady hook: resolve the index, fit vocab, publish runtime dims (before dry-run)."""
        self._plan()

    def _plan(self) -> None:
        if self._planned:
            return
        if not self._live:
            for split in _SPLITS:
                self._seeds_for(split)
            # Stateful transforms fit from their in_key values across the fit splits. The input may be a
            # seed (reader/join output) or produced upstream at item stage (e.g. labelme poly labels), so
            # build the partial workspace up to — but excluding — the fitting node. Dataset-metadata
            # vocabularies opt into all splits via ``fit_all_splits``.
            for rec in self._records:
                node = self._nodes[rec.name]
                if isinstance(node, Transform) and node.stateful:
                    if getattr(node, "fitted", False):
                        continue
                    splits = _SPLITS if node.fit_all_splits else ("train",)
                    in_key = rec.in_key[0]
                    stream = [
                        ws[in_key]
                        for s in splits
                        for seed in (self._base_seeds_for(s) if rec.stage == "plan" else self._seeds.get(s, []))
                        for ws in (dict(seed) if rec.stage == "plan" else self._partial_workspace(seed, rec.name),)
                        if in_key in ws
                    ]
                    node.fit(stream)
        self._republish_dims()
        self._planned = True

    def meta(self) -> dict[str, Any]:
        """Dataset facts (``num_classes``, ``class_names``, ``kpt_shape``, …) for dim-refs and the task contract."""
        self._plan()
        return dict(self._runtime_dims)

    def setup(self, stage: str | None = None) -> None:
        if self._live:
            if stage in (None, "fit"):
                self._stream_reader_node().setup_stream()
            return
        self._plan()

    # -- item stage ---------------------------------------------------------------
    def _item_workspace(self, seed: dict[str, Any], *, augment: bool = True) -> dict[str, Any]:
        """Run the toposorted item subgraph. ``augment=False`` skips the stochastic ``augment``-phase
        nodes; since those are in-place rewrites, their downstream consumers (raster/collate) then read
        the deterministic letterbox values — the eval/report/export slice."""
        workspace: dict[str, Any] = dict(seed)
        for rec in self._item_records:
            if rec.phase == "augment" and not augment:
                continue
            node = self._nodes[rec.name]
            _assign(workspace, rec.out_key, node(*[workspace[k] for k in rec.in_key]))
        return workspace

    def _core_workspace(self, seed: dict[str, Any]) -> dict[str, Any]:
        """Deterministic item workspace (no augments) — eval/report slice and the base a cross-sample
        augment reads its siblings from."""
        return self._item_workspace(seed, augment=False)

    def _sibling_workspace(self, split: str, index: int) -> dict[str, Any]:
        """Cross-sample augment provider: the deterministic core workspace of sibling ``index`` in ``split``
        (no augments → no recursion). Injected onto ``needs_siblings`` augment nodes on the train path."""
        return self._core_workspace(self._seeds_for(split)[index])

    def _bind_siblings(self, split: str | None) -> None:
        """Expose (or clear) the sibling provider on augment nodes that declare ``needs_siblings``."""
        for rec in self._augment_records:
            node = self._nodes[rec.name]
            if getattr(node, "needs_siblings", False):
                node.siblings = partial(self._sibling_workspace, split) if split else None
                node.siblings_len = len(self._seeds_for(split)) if split else 0

    def _partial_workspace(self, seed: dict[str, Any], stop_name: str) -> dict[str, Any]:
        """Item workspace running only nodes topologically before ``stop_name`` — used to fit a
        stateful transform whose input is produced upstream at item stage (not carried on the seed)."""
        workspace: dict[str, Any] = dict(seed)
        for rec in self._item_records:
            if rec.name == stop_name:
                break
            if rec.phase == "augment":
                continue
            node = self._nodes[rec.name]
            if all(k in workspace for k in rec.in_key):
                _assign(workspace, rec.out_key, node(*[workspace[k] for k in rec.in_key]))
        return workspace

    # -- batch stage --------------------------------------------------------------
    def _collate_fn(self, samples: list[dict[str, Any]]) -> Any:
        """Assemble the composed Batch: each collate node builds its field across the sample list.

        A collate record whose in_key is absent from every sample is skipped (the field falls back
        to its optional ``compose_batch`` default) rather than raising — different stream-graph
        seed shapes (a training-buffer seed vs. an eval seed) legitimately carry different field
        subsets from the same declared graph."""
        fields: dict[str, Any] = {}
        for rec in self._collate_records:
            if not all(k in ws for ws in samples for k in rec.in_key):
                continue
            node: Collate = self._nodes[rec.name]
            fields[self._field_names[rec.name]] = node([ws[k] for ws in samples for k in rec.in_key])
        for key, field in self._present_fields.items():
            fields[field] = torch.tensor([ws[key] for ws in samples], dtype=torch.bool)
        return self.batch_type()(**fields)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int = 0) -> Any:
        """Run ``batch``-phase (post-transfer) nodes — GPU transforms / dtype conversion on-device."""
        if not hasattr(batch, "_fields"):
            # Lightning's model summary passes ``example_input_array = (batch,)`` through this hook;
            # unwrap the single composed batch, process it, and re-wrap.
            if isinstance(batch, (tuple, list)) and len(batch) == 1 and hasattr(batch[0], "_fields"):
                return type(batch)([self.on_after_batch_transfer(batch[0], dataloader_idx)])
            return batch
        for rec in self._post_transfer_records:
            node = self._nodes[rec.name]
            values = node(*[getattr(batch, k[len("batch.") :]) for k in rec.in_key])
            batch = batch._replace(**dict(zip((k[len("batch.") :] for k in rec.out_key), _as_tuple(values))))
        return batch

    # -- dataloaders --------------------------------------------------------------
    def _dataloader(self, split: str, *, shuffle: bool, num_workers: int, drop_last: bool) -> DataLoader:
        return DataLoader(
            _ItemDataset(self, split),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and num_workers > 0,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        self._plan()
        if self._live:
            return self._regime.train_iterable(self.resolve_context("policy"))
        self._bind_siblings("train")
        return self._dataloader("train", shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last)

    def _eval_iterable(self) -> Any:
        self.eval_capture_only = True
        return self._regime.eval_iterable(
            self.resolve_context("policy"),
            episodes=self.eval_episodes,
            deterministic=True,
            seed=self._stream_reader_node().seed,
        )

    def val_dataloader(self) -> Any:
        self._plan()
        if self._live:
            return self._eval_iterable()
        if not self._seeds_for("val"):
            return []
        return self._dataloader("val", shuffle=False, num_workers=self.val_num_workers, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        self._plan()
        if self._live:
            return self._eval_iterable()
        return self._dataloader("test", shuffle=False, num_workers=self.val_num_workers, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        self._plan()
        if self._live:
            return self._eval_iterable()
        return self._dataloader(self._predict_split, shuffle=False, num_workers=self.val_num_workers, drop_last=False)

    def bind_infer_source(self, src: Any) -> None:
        bound = False
        for rec in self._readers:
            node = self._nodes[rec.name]
            if hasattr(node, "infer_source"):
                node.infer_source = str(src)
                frames = getattr(node, "_frames", None)
                if isinstance(frames, dict):
                    frames.pop("infer", None)
                bound = True
        if not bound:
            raise RuntimeError("data graph has no reader that supports infer_source.")
        self._predict_split = "infer"
        self._base_seeds.pop("infer", None)
        self._seeds.pop("infer", None)

    # -- pipeline integration -----------------------------------------------------
    def batch_type(self) -> type:
        return compose_batch([*self._field_names.values(), *self._present_fields.values()])

    def batch_meta(self) -> dict[str, tuple[int, ...]]:
        self._plan()
        if self._live:
            fields = set(self._field_names.values())
            return self._regime.batch_meta(self._runtime_dims, fields)
        sample = self._sample_batch()
        meta: dict[str, tuple[int, ...]] = {}
        for field in (*self._field_names.values(), *self._present_fields.values()):
            value = getattr(sample, field)
            if hasattr(value, "shape"):  # tensor fields only; list fields (path/ori_shape/...) are skipped
                meta[field] = tuple(value.shape[1:])
        return meta

    @property
    def schema(self) -> Any:
        self._plan()
        for node in self._nodes.values():
            schema = getattr(node, "schema", None)
            if schema is not None:
                return schema
        raise AttributeError("data graph has no node publishing a tabular schema.")

    @property
    def preprocess_state(self) -> Any:
        self._plan()
        for node in self._nodes.values():
            state = getattr(node, "preprocess_state", None)
            if state is not None:
                return state
        raise AttributeError("data graph has no node publishing tabular preprocess state.")

    def cardinality(self, column: str) -> int:
        return int(self.preprocess_state.cardinality(column))

    def num_features(self, type_slot: str) -> int:
        if type_slot == "temporal":
            return sum(
                len(getattr(self.preprocess_state.columns.get(col), "temporal_features", None) or [])
                for col in self.schema.groups.get("temporal", [])
            )
        return int(self.schema.feature_count(type_slot))

    def _sample_batch(self) -> Any:
        if self._live:
            return self._regime.sample_batch()
        for split in _SPLITS:
            seeds = self._seeds_for(split)
            if seeds:
                return self._collate_fn([self._core_workspace(seeds[0])])
        raise RuntimeError("data graph produced no samples in any split; cannot derive batch_meta.")

    def writer_batch_keys(self) -> list[str]:
        """``batch.*`` fields the writer nodes read at inference (captured alongside the atoms)."""
        return sorted({key for rec in self._writer_records for key in rec.in_key if key.startswith("batch.")})

    def write_predictions(self, *, atoms: dict[str, Any], dst: str | Path) -> list[str]:
        """Hand each writer node the ``capture.*``/``batch.*``/``meta.*`` keys its ``in`` names; returns written paths."""
        self._plan()
        meta = {f"meta.{key}": value for key, value in self._runtime_dims.items()}
        outputs: list[str] = []
        for rec in self._writer_records:
            missing = [key for key in rec.in_key if key not in atoms and key not in meta]
            if missing:
                raise KeyError(f"writer '{rec.name}' reads {missing}, which inference did not produce.")
            inputs = [atoms[key] if key in atoms else meta[key] for key in rec.in_key]
            outputs.extend(str(path) for path in self._nodes[rec.name](*inputs, dst=Path(dst)))
        return outputs

    # -- artifact persistence -----------------------------------------------------
    def save_artifacts(self, directory: str) -> None:
        import pickle

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        for rec in self._records:
            node = self._nodes[rec.name]
            if isinstance(node, Transform) and node.stateful:
                (out / f"{rec.name}.state.pkl").write_bytes(pickle.dumps(node.state()))

    def load_artifacts(self, directory: str, *, required: bool = True) -> None:
        import pickle

        src = Path(directory)
        for rec in self._records:
            node = self._nodes[rec.name]
            if not (isinstance(node, Transform) and node.stateful):
                continue
            path = src / f"{rec.name}.state.pkl"
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"missing fitted state for '{rec.name}': {path}")
                continue
            node.load_state(pickle.loads(path.read_bytes()))
        self._republish_dims()

    def _republish_dims(self) -> None:
        """Collect every node's ``runtime_dims()`` into ``meta()``, expose them as attributes (dim-refs), bind them back."""
        dims: dict[str, Any] = {}
        for node in self._nodes.values():
            dims.update(node.runtime_dims())
        self._runtime_dims = dims
        for name, value in dims.items():
            setattr(self, name, value)
        for node in self._nodes.values():
            node.bind_dims(**dims)
