from __future__ import annotations

import io
import importlib
import json
import os
import pickle
import sys
import tempfile
import torch
import unittest
import warnings
from collections import namedtuple
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


from teia.core.config.scaffold import write_overlay_scaffold
from teia.core.module.net import TeiaNetModule
from teia.base.net import TeiaNode
from teia.core.runtime.composer import compose_project_config
from teia.core.runtime.engine import (
    _prepare_datamodule_config,
    _run_lightning_stage,
    export_project,
    infer_project,
    test_project as run_test_project,
    train_project,
)
from teia.core.runtime.determinism import (
    apply_trainer_determinism,
    configure_torch_determinism,
    raise_enriched_determinism_error,
)
from teia.core.runtime.layout import RuntimeLayout, resolve_runtime_layout
from teia.core.runtime.memory import (
    BYTES_PER_GB,
    MemoryResources,
    RamGuardMonitor,
    configure_ram_guard,
    resolve_ram_guard_bytes,
)
from teia.core.runtime.validate import validate_composed_config
from teia.core.utils import read_yaml

HYDRA_AVAILABLE = importlib.util.find_spec("hydra") is not None

sys.modules.setdefault("tests.test_runtime", sys.modules[__name__])


class SnapshotDataModule:
    def __init__(self, task: str = "generic", data_root: str = "data", project_dir: str | None = None, **_: object) -> None:
        self.task = task
        self.data_root = data_root
        self.project_dir = project_dir

    def ensure_task_metadata(self) -> None:
        return None

    def setup(self, stage: str | None = None) -> None:
        del stage
        return None

    def train_dataloader(self):
        return []

    def val_dataloader(self):
        return []

    def test_dataloader(self):
        return []

    def report_items(self, split: str, project_dir: str | Path | None = None) -> list[dict[str, object]]:
        del split, project_dir
        return []

    def training_reference(self, project_dir: str | Path | None = None, run_dir: str | Path | None = None) -> str:
        del project_dir, run_dir
        return "snapshot-data"


class AlternateSnapshotDataModule(SnapshotDataModule):
    pass


class SnapshotModule:
    def __init__(self, task: str = "generic", token: str = "before-edit", **_: object) -> None:
        self.task = task
        self.token = token
        self.datamodule = None

    def configure_from_datamodule(self, datamodule: object) -> None:
        self.datamodule = datamodule

    def run_teia_stage(
        self,
        *,
        stage: str,
        config: dict,
        datamodule: object,
        run_dir: Path,
        project_dir: Path,
        checkpoint: str | None = None,
    ) -> dict[str, object]:
        del datamodule
        artifact_dir = run_dir / "artifacts" / "snapshot"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{stage}.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "token": self.token,
                    "project_dir": str(project_dir),
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                }
            ),
            encoding="utf-8",
        )
        return {
            "stage": stage,
            "token": self.token,
            "project_dir": str(project_dir),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        }

    def export_teia_bundle(
        self,
        *,
        config: dict,
        run_dir: Path,
        checkpoint: str | None,
        output_dir: Path | None = None,
    ) -> str:
        del config, checkpoint
        bundle_dir = Path(output_dir or (run_dir / "export"))
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "token.txt").write_text(self.token, encoding="utf-8")
        return str(bundle_dir)


class AlternateSnapshotModule(SnapshotModule):
    pass


class EchoNode(TeiaNode):
    def build_module(self, value: float = 0.0, **kwargs) -> None:
        self.value = value
        self.extra = kwargs

    def forward(self, x):
        return x + self.value


class SplitNode(TeiaNode):
    def build_module(self, **kwargs) -> None:
        self.extra = kwargs

    def forward(self, x):
        return x, x + 1


class LearnableNode(TeiaNode):
    def build_module(self, scale: float = 1.0, **kwargs) -> None:
        self.extra = kwargs
        self.weight = torch.nn.Parameter(torch.tensor([float(scale)], dtype=torch.float32))

    def forward(self, x):
        return x * self.weight


class DummyLoss(torch.nn.Module):
    def __init__(self, bias: float = 0.0, **kwargs) -> None:
        super().__init__()
        self.bias = bias
        self.extra = kwargs

    def forward(self, pred, target):
        del target
        return {"loss": pred.sum() * 0 + self.bias}


class DummyPipelineDataModule:
    task = "generic"
    label_names: list = []
    num_labels = 0

    def batch_meta(self) -> dict[str, tuple[int, ...]]:
        return {"x": (4,), "y": (4,)}

    def batch_type(self):
        return namedtuple("Batch", ["x", "y"])


class RuntimeValidationTests(unittest.TestCase):
    @staticmethod
    def _valid_runtime_config() -> dict:
        return {
            "data_root": "data",
            "datamodule": {
                "_target_": "tests.test_runtime.SnapshotDataModule",
                "__task__": "generic",
            },
            "netmodule": {},
            "report": {},
            "trainer": {},
            "runtime": {"determinism": {"mode": "prefer"}},
            "callbacks": {},
            "loggers": {},
            "train": {},
            "test": {},
            "infer": {},
            "export": {},
        }

    def test_pipeline_module_discovers_alias_entries_and_splits_losses(self) -> None:
        module = TeiaNetModule(
            head={
                "_target_": "tests.test_runtime.EchoNode",
                "value": 2.0,
                "in": "feat.backbone",
                "out": {"pred.out": None},
            },
            backbone={
                "_target_": "tests.test_runtime.EchoNode",
                "value": 1.0,
                "in": "batch.x",
                "out": {"feat.backbone": (4,)},
            },
            det_loss={
                "_target_": "tests.test_runtime.DummyLoss",
                "bias": 3.0,
                "in": ["pred.out", "batch.y"],
                "out": {"loss.det": None},
                "weight": 2.5,
            },
        )

        self.assertEqual([rec.name for rec in module._pipeline_records], ["backbone", "head"])
        self.assertEqual(module._pipeline_records[0].out_shape, [(4,)])
        self.assertEqual([rec.name for rec in module._loss_records], ["det_loss"])
        self.assertTrue(module._loss_records[0].is_loss)
        self.assertEqual(module._loss_records[0].weight, 2.5)

    def test_pipeline_module_preserves_out_key_map_order(self) -> None:
        module = TeiaNetModule(
            split={
                "_target_": "tests.test_runtime.SplitNode",
                "in": "batch.x",
                "out": {
                    "feat.first": None,
                    "pred.second": (4,),
                },
            },
        )

        self.assertEqual(module._pipeline_records[0].out_key, ["feat.first", "pred.second"])
        self.assertEqual(module._pipeline_records[0].out_shape, [None, (4,)])

    def test_pipeline_module_requires_valid_producers(self) -> None:
        with self.assertRaisesRegex(ValueError, "in_key 'feat.missing'"):
            TeiaNetModule(
                head={
                    "_target_": "tests.test_runtime.EchoNode",
                    "in": "feat.missing",
                    "out": {"pred.out": None},
                },
            )

    def test_pipeline_module_detects_duplicate_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate out_key 'feat.shared'"):
            TeiaNetModule(
                a={
                    "_target_": "tests.test_runtime.EchoNode",
                    "in": "batch.x",
                    "out": {"feat.shared": None},
                },
                b={
                    "_target_": "tests.test_runtime.EchoNode",
                    "in": "batch.x",
                    "out": {"feat.shared": None},
                },
            )

    def test_pipeline_module_detects_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cycle detected in module graph"):
            TeiaNetModule(
                a={
                    "_target_": "tests.test_runtime.EchoNode",
                    "in": "feat.b",
                    "out": {"feat.a": None},
                },
                b={
                    "_target_": "tests.test_runtime.EchoNode",
                    "in": "feat.a",
                    "out": {"feat.b": None},
                },
            )

    def test_configure_optimizers_returns_legacy_optimizer(self) -> None:
        module = TeiaNetModule(
            optimizer_target="torch.optim.AdamW",
            lr=1.0e-3,
            weight_decay=1.0e-4,
            betas=[0.9, 0.999],
            head={
                "_target_": "tests.test_runtime.LearnableNode",
                "in": "batch.x",
                "out": {"pred.out": None},
            },
        )
        dm = DummyPipelineDataModule()
        module.configure_from_datamodule(dm)

        optimizer = module.configure_optimizers()

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-3)
        self.assertAlmostEqual(optimizer.param_groups[0]["weight_decay"], 1.0e-4)

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra is required for scheduler instantiate tests")
    def test_configure_optimizers_returns_lightning_scheduler_config(self) -> None:
        module = TeiaNetModule(
            optimizer_target="torch.optim.AdamW",
            lr=1.0e-3,
            weight_decay=5.0e-4,
            betas=[0.9, 0.999],
            lr_scheduler={
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.LinearLR",
                    "start_factor": 0.5,
                    "end_factor": 1.0,
                    "total_iters": 3,
                },
                "interval": "epoch",
                "frequency": 1,
                "name": "warmup",
            },
            head={
                "_target_": "tests.test_runtime.LearnableNode",
                "in": "batch.x",
                "out": {"pred.out": None},
            },
        )
        dm = DummyPipelineDataModule()
        module.configure_from_datamodule(dm)

        optim_conf = module.configure_optimizers()

        self.assertIsInstance(optim_conf, dict)
        self.assertIsInstance(optim_conf["optimizer"], torch.optim.AdamW)
        self.assertEqual(optim_conf["lr_scheduler"]["interval"], "epoch")
        self.assertEqual(optim_conf["lr_scheduler"]["frequency"], 1)
        self.assertEqual(optim_conf["lr_scheduler"]["name"], "warmup")
        self.assertIsInstance(optim_conf["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LinearLR)
        self.assertIs(optim_conf["lr_scheduler"]["scheduler"].optimizer, optim_conf["optimizer"])

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra is required for scheduler instantiate tests")
    def test_configure_optimizers_injects_optimizer_into_nested_scheduler_tree(self) -> None:
        module = TeiaNetModule(
            optimizer_target="torch.optim.AdamW",
            lr=1.0e-3,
            weight_decay=5.0e-4,
            betas=[0.9, 0.999],
            lr_scheduler={
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.SequentialLR",
                    "schedulers": [
                        {
                            "_target_": "torch.optim.lr_scheduler.LinearLR",
                            "start_factor": 0.5,
                            "end_factor": 1.0,
                            "total_iters": 3,
                        },
                        {
                            "_target_": "torch.optim.lr_scheduler.LinearLR",
                            "start_factor": 1.0,
                            "end_factor": 0.1,
                            "total_iters": 7,
                        },
                    ],
                    "milestones": [3],
                },
                "interval": "epoch",
                "frequency": 1,
            },
            head={
                "_target_": "tests.test_runtime.LearnableNode",
                "in": "batch.x",
                "out": {"pred.out": None},
            },
        )
        dm = DummyPipelineDataModule()
        module.configure_from_datamodule(dm)

        optim_conf = module.configure_optimizers()
        scheduler = optim_conf["lr_scheduler"]["scheduler"]

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        self.assertEqual(len(scheduler._schedulers), 2)
        self.assertIs(scheduler.optimizer, optim_conf["optimizer"])
        self.assertTrue(all(child.optimizer is optim_conf["optimizer"] for child in scheduler._schedulers))

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra is required for instantiate/configure tests")
    def test_pipeline_module_strips_node_metadata_before_instantiation(self) -> None:
        module = TeiaNetModule(
            head={
                "_target_": "tests.test_runtime.EchoNode",
                "value": 2.0,
                "in": "feat.backbone",
                "out": {"pred.out": None},
            },
            backbone={
                "_target_": "tests.test_runtime.EchoNode",
                "value": 1.0,
                "in": "batch.x",
                "out": {"feat.backbone": None},
            },
            det_loss={
                "_target_": "tests.test_runtime.DummyLoss",
                "bias": 3.0,
                "in": ["pred.out", "batch.y"],
                "weight": 1.0,
            },
        )

        module.configure_from_datamodule(DummyPipelineDataModule())

        self.assertEqual(module.backbone.extra, {})
        self.assertEqual(module.head.extra, {})
        self.assertEqual(module.det_loss.extra, {})

    def test_train_suppresses_intentional_eval_mode_warning_only_when_model_allows_it(self) -> None:
        class FakeTrainer:
            precision = "32-true"

            def __init__(self, **kwargs):
                del kwargs

            def fit(self, model, datamodule, ckpt_path=None):
                del model, datamodule, ckpt_path
                warnings.warn(
                    "Found 217 module(s) in eval mode at the start of training. "
                    "This may lead to unexpected behavior during training. If this is intentional, you can ignore this warning.",
                    UserWarning,
                )
                warnings.warn("unrelated warning", UserWarning)
                return "fit-ok"

        fake_lightning = ModuleType("lightning.pytorch")
        fake_lightning.Trainer = FakeTrainer

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightning.pytorch":
                return fake_lightning
            return real_import(name, globals, locals, fromlist, level)

        class FakeModel:
            def configure_from_datamodule(self, datamodule: object) -> None:
                del datamodule

            def allow_eval_modules_at_train_start(self) -> bool:
                return True

        with patch("builtins.__import__", side_effect=fake_import):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _run_lightning_stage(
                    stage="train",
                    config={"trainer": {}, "callbacks": {"items": []}, "loggers": {"items": []}, "train": {}},
                    datamodule_obj=object(),
                    module_obj=FakeModel(),
                    checkpoint=None,
                    run_dir=Path("."),
                )

        self.assertEqual(result["result"], "fit-ok")
        self.assertEqual([str(item.message) for item in caught], ["unrelated warning"])

    def _make_layout(root: Path) -> RuntimeLayout:
        run_dir = root / "runs" / "manual"
        return RuntimeLayout(
            project_dir=root,
            runs_root=root / "runs",
            cache_root=root / ".cache",
            run_name="manual",
            run_dir=run_dir,
            config_dir=run_dir / "config",
            logs_dir=run_dir / "logs",
            eval_dir=run_dir / "eval",
            infer_dir=run_dir / "infer",
            export_dir=run_dir / "export",
            artifacts_dir=run_dir / "artifacts",
        )

    @staticmethod
    def _write_snapshot_test_project(root: Path, token: str = "before-edit") -> None:
        write_overlay_scaffold(root=root, force=True)
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "conf" / "evalmodule").mkdir(parents=True, exist_ok=True)
        (root / "conf" / "datamodule").mkdir(parents=True, exist_ok=True)
        (root / "conf" / "netmodule").mkdir(parents=True, exist_ok=True)
        (root / "conf" / "datamodule" / "custom.yaml").write_text(
            "_target_: tests.test_runtime.SnapshotDataModule\n"
            "__task__: snapshot-task\n"
            "task: snapshot-task\n",
            encoding="utf-8",
        )
        (root / "conf" / "netmodule" / "custom.yaml").write_text(
            "_target_: tests.test_runtime.SnapshotModule\n"
            "__task__: snapshot-task\n"
            "task: snapshot-task\n"
            f"token: {token}\n",
            encoding="utf-8",
        )
        (root / "conf" / "evalmodule" / "custom.yaml").write_text(
            "enabled: false\n",
            encoding="utf-8",
        )

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra is required for config composition tests")
    def test_runtime_ram_guard_defaults_compose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = compose_project_config(project_dir=Path(tmp), overrides=[])

        ram_guard = config["runtime"]["ram_guard"]
        self.assertEqual(config["runtime"]["determinism"]["mode"], "prefer")
        self.assertEqual(ram_guard["type"], "total_percent")
        self.assertIsNone(ram_guard["gb"])
        self.assertEqual(ram_guard["percent"], 0.8)
        self.assertEqual(ram_guard["warn_fraction"], 0.9)

    def test_determinism_modes_configure_torch_and_lightning(self) -> None:
        original_enabled = torch.are_deterministic_algorithms_enabled()
        original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        original_benchmark = torch.backends.cudnn.benchmark
        original_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        cases = [
            ("force", True, False, True),
            ("prefer", True, True, "warn"),
            ("stochastic", False, False, False),
        ]
        try:
            for mode, enabled, warn_only, lightning_value in cases:
                with self.subTest(mode=mode):
                    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
                    config = {"runtime": {"determinism": {"mode": mode}}}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        configure_torch_determinism(config)
                    trainer_kwargs: dict[str, object] = {}
                    apply_trainer_determinism(config, trainer_kwargs)

                    self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                    self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn_only)
                    self.assertEqual(trainer_kwargs["deterministic"], lightning_value)
                    if enabled:
                        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                        self.assertFalse(trainer_kwargs["benchmark"])
                        self.assertFalse(torch.backends.cudnn.benchmark)
                    else:
                        self.assertNotIn("benchmark", trainer_kwargs)
                        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)
        finally:
            torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)
            torch.backends.cudnn.benchmark = original_benchmark
            if original_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_workspace

    def test_prefer_determinism_rewrites_torch_warn_only_warning(self) -> None:
        original_enabled = torch.are_deterministic_algorithms_enabled()
        original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        original_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            configure_torch_determinism({"runtime": {"determinism": {"mode": "prefer"}}})
            stderr = io.StringIO()
            message = (
                "unit_test_warn_only_op does not have a deterministic implementation, but you set "
                "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at "
                "https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support."
            )
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                with redirect_stderr(stderr):
                    warnings.warn(message, UserWarning)

            rendered = stderr.getvalue()
            self.assertIn("[teia] determinism warning:", rendered)
            self.assertIn("unit_test_warn_only_op", rendered)
            self.assertIn("runtime.determinism.mode=force", rendered)
            self.assertIn("runtime.determinism.mode=stochastic", rendered)
            self.assertNotIn("pytorch/pytorch/issues", rendered)
        finally:
            configure_torch_determinism({"runtime": {"determinism": {"mode": "stochastic"}}})
            torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)
            if original_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = original_workspace

    def test_determinism_validation_rejects_component_controls(self) -> None:
        config = self._valid_runtime_config()
        config["trainer"]["deterministic"] = True
        with self.assertRaisesRegex(ValueError, "runtime.determinism.mode"):
            validate_composed_config(config)

        config = self._valid_runtime_config()
        config["trainer"]["benchmark"] = True
        with self.assertRaisesRegex(ValueError, "trainer.benchmark=true"):
            validate_composed_config(config)

    def test_force_determinism_error_includes_mode_suggestions(self) -> None:
        source = RuntimeError("nll_loss2d_forward_out_cuda_template does not have a deterministic implementation")
        with self.assertRaises(RuntimeError) as caught:
            raise_enriched_determinism_error({"runtime": {"determinism": {"mode": "force"}}}, source)

        message = str(caught.exception)
        self.assertIn("runtime.determinism.mode=prefer", message)
        self.assertIn("runtime.determinism.mode=stochastic", message)
        self.assertIs(caught.exception.__cause__, source)

    def test_ram_guard_resolver_routes_modes(self) -> None:
        resources = MemoryResources(total_bytes=16 * BYTES_PER_GB, available_bytes=8 * BYTES_PER_GB)

        self.assertEqual(resolve_ram_guard_bytes({"type": "gb", "gb": 4}, resources), 4 * BYTES_PER_GB)
        self.assertEqual(resolve_ram_guard_bytes({"type": "total_percent", "percent": 0.5}, resources), 8 * BYTES_PER_GB)
        self.assertEqual(resolve_ram_guard_bytes({"type": "usable_percent", "percent": 0.5}, resources), 4 * BYTES_PER_GB)

    def test_ram_guard_resolver_rejects_invalid_config(self) -> None:
        resources = MemoryResources(total_bytes=16 * BYTES_PER_GB, available_bytes=8 * BYTES_PER_GB)

        cases = [
            ({"type": "bad"}, "Invalid runtime.ram_guard.type"),
            ({"type": "gb"}, "runtime.ram_guard.gb is required"),
            ({"type": "gb", "gb": 0}, "runtime.ram_guard.gb must be > 0"),
            ({"type": "usable_percent", "percent": 1.5}, "runtime.ram_guard.percent must be <= 1"),
            ({"type": "total_percent", "percent": 0}, "runtime.ram_guard.percent must be > 0"),
        ]
        for guard_cfg, pattern in cases:
            with self.subTest(guard_cfg=guard_cfg):
                with self.assertRaisesRegex(ValueError, pattern):
                    resolve_ram_guard_bytes(guard_cfg, resources)

        with patch(
            "teia.core.runtime.memory.read_memory_resources",
            return_value=MemoryResources(total_bytes=1, available_bytes=1),
        ):
            with self.assertRaisesRegex(ValueError, "non-positive bytes"):
                configure_ram_guard(
                    {"runtime": {"ram_guard": {"type": "usable_percent", "percent": 0.1, "warn_fraction": 0.9}}}
                )

    def test_ram_guard_none_skips_monitor(self) -> None:
        with patch("teia.core.runtime.memory.start_ram_guard") as start_guard:
            self.assertIsNone(configure_ram_guard({"runtime": {"ram_guard": {"type": "none"}}}))

        start_guard.assert_not_called()

    def test_ram_guard_configure_starts_monitor(self) -> None:
        with patch("teia.core.runtime.memory.read_memory_resources") as read_resources:
            with patch("teia.core.runtime.memory.start_ram_guard") as start_guard:
                result = configure_ram_guard(
                    {"runtime": {"ram_guard": {"type": "gb", "gb": 4, "warn_fraction": 0.9}}}
                )

        read_resources.assert_called_once()
        start_guard.assert_called_once_with(4 * BYTES_PER_GB, 0.9)
        self.assertEqual(result, 4 * BYTES_PER_GB)

    def test_ram_guard_monitor_warns_every_ten_seconds_near_limit(self) -> None:
        messages: list[str] = []
        monitor = RamGuardMonitor(
            limit_bytes=100,
            warn_fraction=0.9,
            read_current_bytes=lambda: 95,
            emit=messages.append,
        )

        self.assertTrue(monitor.check(now=0))
        self.assertFalse(monitor.check(now=5))
        self.assertTrue(monitor.check(now=10))
        self.assertEqual(len(messages), 2)
        self.assertIn("[teia] RAM guard warning:", messages[0])

    def test_ram_guard_monitor_aborts_when_limit_crossed(self) -> None:
        messages: list[str] = []
        aborted: list[bool] = []
        monitor = RamGuardMonitor(
            limit_bytes=100,
            warn_fraction=0.9,
            read_current_bytes=lambda: 100,
            emit=messages.append,
            abort=lambda: aborted.append(True),
        )

        self.assertTrue(monitor.check(now=0))
        self.assertEqual(aborted, [True])
        self.assertIn("[teia] RAM guard tripped:", messages[0])

    def test_stage_commands_configure_ram_guard_before_instantiation(self) -> None:
        import teia.core.runtime.engine as runtime_engine

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")
            real_instantiate = runtime_engine.instantiate
            events: list[str] = []

            def mark_ram_guard(config):
                del config
                events.append("memory")
                return None

            def mark_instantiate(payload):
                events.append("instantiate")
                return real_instantiate(payload)

            with patch("teia.core.runtime.engine.configure_ram_guard", side_effect=mark_ram_guard):
                with patch("teia.core.runtime.engine.instantiate", side_effect=mark_instantiate):
                    train_context = train_project(
                        project_dir=root,
                        overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                        run_name="snapshot-run",
                        skip_post_test=True,
                        skip_report=True,
                        skip_export=True,
                    )
            self.assertLess(events.index("memory"), events.index("instantiate"))

            run_dir = Path(train_context["run_dir"])
            # `infer` is dataset-only (Lightning predict); the generic snapshot module is not a
            # dataset datamodule, so it is excluded here — test/export share the same early ordering.
            for stage, call in [
                ("test", lambda: run_test_project(run_dir=run_dir, skip_report=True)),
                ("export", lambda: export_project(run_dir=run_dir)),
            ]:
                with self.subTest(stage=stage):
                    events.clear()
                    with patch("teia.core.runtime.engine.configure_ram_guard", side_effect=mark_ram_guard):
                        with patch("teia.core.runtime.engine.instantiate", side_effect=mark_instantiate):
                            call()
                    self.assertLess(events.index("memory"), events.index("instantiate"))

    @unittest.skipUnless(HYDRA_AVAILABLE, "hydra is required for config composition tests")
    def test_compose_project_config_does_not_warn_for_hydra_group_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                config = compose_project_config(
                    project_dir=root,
                    overrides=[
                        "hydra/job_logging=default",
                    ],
                )

            self.assertFalse(
                any("possible typo" in str(item.message) for item in caught),
                msg=[str(item.message) for item in caught],
            )
            del config

    def test_lightning_trainer_does_not_receive_internal_metadata(self) -> None:
        captured: dict[str, object] = {}

        class FakeTrainer:
            precision = "32-true"

            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit(self, model, datamodule, ckpt_path=None):
                return {"model": model, "datamodule": datamodule, "ckpt_path": ckpt_path}

        fake_lightning = ModuleType("lightning.pytorch")
        fake_lightning.Trainer = FakeTrainer

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightning.pytorch":
                return fake_lightning
            return real_import(name, globals, locals, fromlist, level)

        model = SimpleNamespace(configure_from_datamodule=lambda _: None)
        datamodule = object()
        config = {
            "trainer": {
                "accelerator": "auto",
                "devices": 1,
                "__group__": "trainer",
                "__config_name__": "default",
            },
            "callbacks": {"items": []},
            "loggers": {"items": []},
            "train": {"resume_from": None},
        }

        with patch("builtins.__import__", side_effect=fake_import):
            _run_lightning_stage(
                stage="train",
                config=config,
                datamodule_obj=datamodule,
                module_obj=model,
                checkpoint=None,
                run_dir=Path("."),
            )

        self.assertEqual(captured["accelerator"], "auto")
        self.assertEqual(captured["devices"], 1)
        self.assertNotIn("__group__", captured)
        self.assertNotIn("__config_name__", captured)

    def test_continued_train_restores_weights_without_resuming_trainer_state(self) -> None:
        captured: dict[str, object] = {}

        class FakeTrainer:
            precision = "32-true"

            def __init__(self, **kwargs):
                del kwargs

            def fit(self, model, datamodule, ckpt_path=None):
                captured["model"] = model
                captured["datamodule"] = datamodule
                captured["ckpt_path"] = ckpt_path
                return "fit-ok"

        fake_lightning = ModuleType("lightning.pytorch")
        fake_lightning.Trainer = FakeTrainer

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightning.pytorch":
                return fake_lightning
            return real_import(name, globals, locals, fromlist, level)

        model = SimpleNamespace(configure_from_datamodule=lambda _: None)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch("teia.core.runtime.engine._restore_model_weights") as restore_weights:
                result = _run_lightning_stage(
                    stage="train",
                    config={"trainer": {}, "callbacks": {"items": []}, "loggers": {"items": []}, "train": {}},
                    datamodule_obj=object(),
                    module_obj=model,
                    checkpoint="runs/parent/checkpoints/best.ckpt",
                    run_dir=Path("."),
                    restore_weights_only=True,
                )

        restore_weights.assert_called_once_with(model, "runs/parent/checkpoints/best.ckpt")
        self.assertEqual(captured["ckpt_path"], None)
        self.assertEqual(result["result"], "fit-ok")

    def test_fresh_train_keeps_checkpoint_resume_path(self) -> None:
        captured: dict[str, object] = {}

        class FakeTrainer:
            precision = "32-true"

            def __init__(self, **kwargs):
                del kwargs

            def fit(self, model, datamodule, ckpt_path=None, weights_only=None):
                del model, datamodule
                captured["ckpt_path"] = ckpt_path
                captured["weights_only"] = weights_only
                return "fit-ok"

        fake_lightning = ModuleType("lightning.pytorch")
        fake_lightning.Trainer = FakeTrainer

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightning.pytorch":
                return fake_lightning
            return real_import(name, globals, locals, fromlist, level)

        model = SimpleNamespace(configure_from_datamodule=lambda _: None)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch("teia.core.runtime.engine._restore_model_weights") as restore_weights:
                _run_lightning_stage(
                    stage="train",
                    config={
                        "trainer": {},
                        "callbacks": {"items": []},
                        "loggers": {"items": []},
                        "train": {"resume_from": "runs/current/checkpoints/last.ckpt"},
                    },
                    datamodule_obj=object(),
                    module_obj=model,
                    checkpoint=None,
                    run_dir=Path("."),
                    restore_weights_only=False,
                )

        restore_weights.assert_not_called()
        self.assertEqual(captured["ckpt_path"], "runs/current/checkpoints/last.ckpt")
        self.assertEqual(captured["weights_only"], False)

    def test_test_stage_restores_checkpoints_with_trusted_load_mode(self) -> None:
        # The "test" stage manually restores weights (EMA-averaged state_dict) up front and calls
        # trainer.test(ckpt_path=None) so Lightning doesn't also try to restore callback state — see
        # the comment above the `stage == "test"` branch in `_run_lightning_stage`.
        captured: dict[str, object] = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                del kwargs

            def test(self, model, datamodule, ckpt_path=None):
                del model, datamodule
                captured["ckpt_path"] = ckpt_path
                return "test-ok"

        fake_lightning = ModuleType("lightning.pytorch")
        fake_lightning.Trainer = FakeTrainer

        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lightning.pytorch":
                return fake_lightning
            return real_import(name, globals, locals, fromlist, level)

        model = SimpleNamespace(configure_from_datamodule=lambda _: None)
        restored: dict[str, object] = {}

        def fake_restore_model_weights(module_obj, checkpoint):
            restored["module"] = module_obj
            restored["checkpoint"] = checkpoint

        with patch("builtins.__import__", side_effect=fake_import):
            with patch(
                "teia.core.runtime.engine._restore_model_weights",
                side_effect=fake_restore_model_weights,
            ):
                result = _run_lightning_stage(
                    stage="test",
                    config={"trainer": {}, "callbacks": {"items": []}, "loggers": {"items": []}, "test": {}},
                    datamodule_obj=object(),
                    module_obj=model,
                    checkpoint="runs/current/checkpoints/best.ckpt",
                    run_dir=Path("."),
                )

        self.assertEqual(result["result"], "test-ok")
        self.assertEqual(restored["checkpoint"], "runs/current/checkpoints/best.ckpt")
        self.assertIsNone(captured["ckpt_path"])

    def test_runtime_layout_uses_root_level_names_and_run_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = compose_project_config(
                project_dir=root,
                overrides=[
                    "project_name=teia",
                    "experiment_name=det-v2",
                    "run_parts=[project,teia,experiment,run]",
                ],
            )

            layout = resolve_runtime_layout(config=config, project_dir=root, cli_run_name="manual-run")

            self.assertEqual(layout.run_name, "manual-run")
            self.assertEqual(layout.run_dir, root / "runs" / "teia" / "teia" / "det-v2" / "manual-run")

    def test_project_dir_is_importable_before_the_task_contract_is_validated(self) -> None:
        class Stop(Exception):
            pass

        seen: dict[str, bool] = {}

        def probe(config: dict) -> None:
            seen["on_path"] = str(root.resolve()) in sys.path
            raise Stop

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                with patch("teia.core.runtime.engine.validate_composed_config", probe):
                    with self.assertRaises(Stop):
                        train_project(project_dir=root, overrides=["task=tabular-reg", "data_root=."])
            finally:
                sys.path[:] = [entry for entry in sys.path if entry != str(root.resolve())]

        self.assertTrue(seen["on_path"])

    def test_prepare_datamodule_config_injects_supported_runtime_kwargs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datamodule" / "examples" / "multi-cls").mkdir(parents=True, exist_ok=True)
            prepared = _prepare_datamodule_config(
                config={
                    "data_root": "datamodule/examples/multi-cls",
                    "datamodule": {"_target_": "teia.core.datamodule.TeiaDataModule"},
                },
                project_dir=root,
            )

            self.assertEqual(prepared["project_dir"], str(root))
            self.assertEqual(prepared["data_root"], str((root / "datamodule/examples/multi-cls").resolve()))

    def test_prepare_datamodule_config_rejects_targets_without_data_root_support(self) -> None:
        class LegacyDataModule:
            def __init__(self, root: str = "legacy") -> None:
                self.root = root

        fake_module = ModuleType("teia.vision.datamodule.legacy_test_module")
        fake_module.LegacyDataModule = LegacyDataModule

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules,
            {"teia.vision.datamodule.legacy_test_module": fake_module},
        ):
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "does not accept `data_root`"):
                _prepare_datamodule_config(
                    config={
                        "data_root": "legacy",
                        "datamodule": {"_target_": "teia.vision.datamodule.legacy_test_module.LegacyDataModule"},
                    },
                    project_dir=root,
                )

    def test_legacy_top_level_data_group_is_rejected(self) -> None:
        config = {
            "datamodule": {"_target_": "teia.vision.datamodule.VisionDataModule"},
            "netmodule": {},
            "report": {},
            "trainer": {},
            "runtime": {},
            "callbacks": {},
            "loggers": {},
            "export": {},
            "train": {},
            "test": {},
            "infer": {},
            "data": {},
        }
        with self.assertRaisesRegex(ValueError, "Legacy top-level config group 'data'"):
            validate_composed_config(config)

    def test_train_snapshots_copy_conf_tree_and_test_uses_saved_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            train_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="snapshot-run",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            run_dir = Path(train_context["run_dir"])

            self.assertTrue((run_dir / "config" / "conf" / "netmodule" / "custom.yaml").exists())
            self.assertTrue((run_dir / "config" / "composed.yaml").exists())
            self.assertTrue((run_dir / "config" / "overrides.yaml").exists())
            self.assertTrue((run_dir / "config" / "hydra.yaml").exists())

            (root / "conf" / "netmodule" / "custom.yaml").write_text(
                "_target_: tests.test_runtime.SnapshotModule\n"
                "__task__: snapshot-task\n"
                "task: snapshot-task\n"
                "token: after-edit\n",
                encoding="utf-8",
            )

            test_context = run_test_project(run_dir=run_dir, skip_report=True)

            self.assertEqual(test_context["result"]["token"], "before-edit")
            stage_payload = json.loads((run_dir / "artifacts" / "stages" / "test.json").read_text(encoding="utf-8"))
            self.assertEqual(stage_payload["stage"], "test")
            self.assertEqual(stage_payload["run_dir"], str(run_dir))

    def test_train_does_not_export_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            train_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="snapshot-run",
                skip_post_test=True,
                skip_report=True,
            )
            run_dir = Path(train_context["run_dir"])

            self.assertNotIn("post_export", train_context)
            self.assertFalse((run_dir / "artifacts" / "stages" / "export.json").exists())
            self.assertFalse((run_dir / "export" / "token.txt").exists())

    def test_train_can_opt_in_to_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            train_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="snapshot-run",
                skip_post_test=True,
                skip_report=True,
                skip_export=False,
            )
            run_dir = Path(train_context["run_dir"])

            self.assertIn("post_export", train_context)
            self.assertTrue((run_dir / "artifacts" / "stages" / "export.json").exists())
            self.assertEqual((run_dir / "export" / "token.txt").read_text(encoding="utf-8"), "before-edit")

    def test_continued_train_uses_saved_snapshot_and_combined_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            phase1_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="phase-one",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            phase1_run_dir = Path(phase1_context["run_dir"])

            (root / "conf" / "netmodule" / "custom.yaml").write_text(
                "_target_: tests.test_runtime.SnapshotModule\n"
                "__task__: snapshot-task\n"
                "task: snapshot-task\n"
                "token: after-edit\n",
                encoding="utf-8",
            )

            phase2_context = train_project(
                from_run_dir=phase1_run_dir,
                overrides=["experiment_name=phase-two"],
                run_name="phase-two",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            phase2_run_dir = Path(phase2_context["run_dir"])

            self.assertNotEqual(phase2_run_dir, phase1_run_dir)
            self.assertEqual(phase2_context["result"]["token"], "before-edit")
            self.assertEqual(phase2_context["train_mode"], "continued")
            self.assertEqual(phase2_context["parent_run_dir"], str(phase1_run_dir))

            overrides_payload = read_yaml(phase2_run_dir / "config" / "overrides.yaml")
            self.assertEqual(
                overrides_payload["overrides"],
                ["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data", "experiment_name=phase-two"],
            )

            stage_payload = json.loads((phase2_run_dir / "artifacts" / "stages" / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(stage_payload["train_mode"], "continued")
            self.assertEqual(stage_payload["parent_run_dir"], str(phase1_run_dir))
            self.assertTrue((phase2_run_dir / "config" / "conf" / "netmodule" / "custom.yaml").exists())

    def test_continued_train_checkpoint_policy_prefers_parent_run_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            phase1_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="phase-one",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            phase1_run_dir = Path(phase1_context["run_dir"])
            best_ckpt = phase1_run_dir / "checkpoints" / "best.ckpt"
            last_ckpt = phase1_run_dir / "checkpoints" / "last.ckpt"
            manual_ckpt = phase1_run_dir / "weights" / "manual.ckpt"
            for checkpoint_path in (best_ckpt, last_ckpt, manual_ckpt):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.write_text("checkpoint", encoding="utf-8")

            stage_payload_path = phase1_run_dir / "artifacts" / "stages" / "train.json"
            stage_payload = json.loads(stage_payload_path.read_text(encoding="utf-8"))
            stage_payload["report_checkpoint"] = str(best_ckpt)
            stage_payload["best_checkpoint"] = str(best_ckpt)
            stage_payload["last_checkpoint"] = str(last_ckpt)
            stage_payload_path.write_text(json.dumps(stage_payload, indent=2, sort_keys=True), encoding="utf-8")

            default_context = train_project(
                from_run_dir=phase1_run_dir,
                overrides=["experiment_name=phase-two-default"],
                run_name="phase-two-default",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            last_context = train_project(
                from_run_dir=phase1_run_dir,
                from_run_weights="last",
                overrides=["experiment_name=phase-two-last"],
                run_name="phase-two-last",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            explicit_context = train_project(
                from_run_dir=phase1_run_dir,
                checkpoint=str(manual_ckpt),
                overrides=["experiment_name=phase-two-explicit"],
                run_name="phase-two-explicit",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )

            self.assertEqual(default_context["parent_checkpoint"], str(best_ckpt))
            self.assertEqual(default_context["result"]["checkpoint"], str(best_ckpt))
            self.assertEqual(last_context["parent_checkpoint"], str(last_ckpt))
            self.assertEqual(last_context["result"]["checkpoint"], str(last_ckpt))
            self.assertEqual(explicit_context["parent_checkpoint"], str(manual_ckpt))
            self.assertEqual(explicit_context["result"]["checkpoint"], str(manual_ckpt))

    def test_continued_train_rejects_resume_from_and_contract_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            phase1_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="phase-one",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            phase1_run_dir = Path(phase1_context["run_dir"])

            with self.assertRaisesRegex(ValueError, "cannot change datamodule\\._target_"):
                train_project(
                    from_run_dir=phase1_run_dir,
                    overrides=["datamodule._target_=tests.test_runtime.AlternateSnapshotDataModule"],
                    run_name="bad-datamodule",
                    skip_post_test=True,
                    skip_report=True,
                    skip_export=True,
                )

            with self.assertRaisesRegex(ValueError, "cannot change netmodule\\._target_"):
                train_project(
                    from_run_dir=phase1_run_dir,
                    overrides=["netmodule._target_=tests.test_runtime.AlternateSnapshotModule"],
                    run_name="bad-module",
                    skip_post_test=True,
                    skip_report=True,
                    skip_export=True,
                )

            with self.assertRaisesRegex(ValueError, "does not support train.resume_from"):
                train_project(
                    from_run_dir=phase1_run_dir,
                    overrides=["train.resume_from=/tmp/legacy.ckpt"],
                    run_name="bad-resume",
                    skip_post_test=True,
                    skip_report=True,
                    skip_export=True,
                )

    def test_export_rehydrates_saved_snapshot_and_infer_is_dataset_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            train_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="snapshot-run",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            run_dir = Path(train_context["run_dir"])

            (root / "conf" / "netmodule" / "custom.yaml").write_text(
                "_target_: tests.test_runtime.SnapshotModule\n"
                "__task__: snapshot-task\n"
                "task: snapshot-task\n"
                "token: after-edit\n",
                encoding="utf-8",
            )

            # `teia infer` is dataset-only: a datamodule without `write_predictions` fails fast.
            with self.assertRaises(RuntimeError):
                infer_project(run_dir=run_dir)

            export_context = export_project(run_dir=run_dir)
            export_dir = Path(export_context["result"])
            self.assertTrue((export_dir / "token.txt").exists())
            self.assertEqual((export_dir / "token.txt").read_text(encoding="utf-8"), "before-edit")

    def test_train_and_export_emit_phase_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            train_stdout = io.StringIO()
            with redirect_stdout(train_stdout):
                train_context = train_project(
                    project_dir=root,
                    overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                    run_name="snapshot-run",
                    skip_post_test=True,
                    skip_report=True,
                    skip_export=True,
                )

            train_output = train_stdout.getvalue()
            self.assertIn("[teia] fit start:", train_output)
            self.assertIn("[teia] fit complete:", train_output)
            self.assertIn("mode=fresh", train_output)

            export_stdout = io.StringIO()
            with redirect_stdout(export_stdout):
                export_project(run_dir=Path(train_context["run_dir"]))

            export_output = export_stdout.getvalue()
            self.assertIn("[teia] export start:", export_output)
            self.assertIn("[teia] export complete:", export_output)

    def test_test_project_always_runs_report(self) -> None:
        """``test_project`` has no pre-flight capture-existence gate at all any more.

        Part 4c finished migrating every vision task (det/obb/inst-seg/pose/sem-seg/gen, joining
        cls/multi-cls from Part 1) off the legacy pickle onto the route-keyed store, and the
        Part 4 cleanup pass deleted the now-fully-dead pre-flight check
        (`_require_vision_report_capture`, `_is_image_shaped`, `core/predictions/artifacts.py`)
        along with it — "no task-gate replaces it ... this existence check is simply removed",
        the same reasoning Part 1 already applied to cls/multi-cls.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_snapshot_test_project(root, token="before-edit")

            train_context = train_project(
                project_dir=root,
                overrides=["datamodule=custom", "netmodule=custom", "evalmodule=custom", "data_root=data"],
                run_name="snapshot-run",
                skip_post_test=True,
                skip_report=True,
                skip_export=True,
            )
            run_dir = Path(train_context["run_dir"])

            calls: list[str] = []

            def fake_run_report(config, context):
                del config, context
                calls.append("report")
                return []

            with patch("teia.core.runtime.engine.run_offline_eval", side_effect=fake_run_report):
                run_test_project(run_dir=run_dir, skip_report=False)

            self.assertEqual(calls, ["report"])


if __name__ == "__main__":
    unittest.main()
