from __future__ import annotations

import contextlib
import inspect
import logging
import random
import warnings
from pathlib import Path
from typing import Any

from teia.base.protocols import (
    ArtifactPersistable,
    BatchMetaSource,
    TeiaStageModule,
    InferSourceBindable,
    NativeStageRunner,
    TaskMetadataReady,
)
from teia.core.deps import raise_with_dependency_context
from teia.core.export.onnx import export_bundle
from teia.core.infer.runner import run_infer
from teia.core.instantiate import instantiate, instantiate_many
from teia.core.torch_compat import load_trusted_checkpoint, register_safe_checkpoint_globals
from teia.core.capture.callback import CaptureCallback
from teia.core.eval.graph import build_eval_graph, discover_node_entries
from teia.core.eval.runner import required_splits, run_offline_eval, split_evalmodule
from teia.core.capture.walltime import WalltimeCallback
from teia.core.runtime.run_descriptor import finish_run_stage, start_run_descriptor
from teia.core.runtime.composer import (
    compose_overlay_config,
    compose_run_config,
    compose_run_config_tree,
    load_saved_composed_config,
    resolve_resumable_run_dir,
)
from teia.core.runtime.determinism import (
    apply_trainer_determinism,
    configure_torch_determinism,
    raise_enriched_determinism_error,
)
from teia.core.runtime.hooks import run_hook_group
from teia.core.runtime.layout import RuntimeLayout, resolve_runtime_layout
from teia.core.runtime.memory import configure_ram_guard
from teia.core.runtime.project_python import load_project_python
from teia.core.runtime.validate import validate_composed_config
from teia.core.utils import (
    copy_tree,
    disable_pretrained_loading,
    ensure_dir,
    teia_log,
    import_string,
    read_json,
    strip_internal_metadata,
    write_json,
    write_yaml,
)


def train_project(
    project_dir: Path | None = None,
    config_dir: Path | None = None,
    overrides: list[str] | None = None,
    run_name: str | None = None,
    checkpoint: str | None = None,
    from_run_dir: Path | None = None,
    from_run_weights: str = "best",
    skip_report: bool = False,
    skip_export: bool | None = None,
    skip_post_test: bool = False,
) -> dict[str, Any]:
    (
        config,
        hydra_payload,
        resolved_project_dir,
        source_conf_root,
        resolved_overrides,
        lineage,
        initial_checkpoint,
    ) = _prepare_train_bootstrap(
        project_dir=project_dir,
        config_dir=config_dir,
        overrides=overrides or [],
        checkpoint=checkpoint,
        from_run_dir=from_run_dir,
        from_run_weights=from_run_weights,
    )
    load_project_python(resolved_project_dir)
    validate_composed_config(config)
    configure_ram_guard(config)

    context = _prepare_train_context(
        project_dir=resolved_project_dir,
        config=config,
        hydra_payload=hydra_payload,
        source_conf_root=source_conf_root,
        run_name=run_name,
        overrides=resolved_overrides,
    )
    context.update(lineage)
    start_run_descriptor(run_dir=context["run_dir"], config=config)
    _seed_runtime(config)
    _configure_torch_runtime(config)
    _persist_train_snapshots(context)

    _with_dependency_context(config, run_hook_group, config, "pre_build", context)
    train_checkpoint = initial_checkpoint or (config.get("train") or {}).get("resume_from")
    datamodule_obj = _with_dependency_context(
        config,
        instantiate,
        _prepare_datamodule_config(config=config, project_dir=resolved_project_dir),
    )
    module_obj = _with_dependency_context(config, instantiate, _prepare_module_config(config=config, checkpoint=train_checkpoint))
    context["datamodule"] = datamodule_obj
    context["module"] = module_obj
    context["model"] = module_obj

    _with_dependency_context(config, run_hook_group, config, "pre_train", context)
    teia_log(
        "fit start",
        run_dir=context["run_dir"],
        mode=context.get("train_mode"),
        checkpoint=train_checkpoint,
        restore_weights_only=bool(from_run_dir is not None),
    )
    stage_result = _with_dependency_context(
        config,
        _run_stage_module,
        stage="train",
        config=config,
        datamodule_obj=datamodule_obj,
        module_obj=module_obj,
        checkpoint=initial_checkpoint,
        run_dir=context["run_dir"],
        project_dir=resolved_project_dir,
        restore_weights_only=bool(from_run_dir is not None),
    )
    context.update(stage_result)
    context["result"] = stage_result["result"]
    context["report_checkpoint"] = _resolve_checkpoint(stage="train", checkpoint=None, config=config, context=context)
    teia_log(
        "fit complete",
        run_dir=context["run_dir"],
        checkpoint=context.get("report_checkpoint"),
    )
    _with_dependency_context(config, run_hook_group, config, "post_train", context)
    _write_stage_metadata(context=context, stage="train")
    finish_run_stage(run_dir=context["run_dir"], stage="train", routes=_module_routes(module_obj))
    del stage_result  # drop the extra trainer ref so cleanup can free optimizer GPU state

    train_cfg = config.get("train") or {}
    if not skip_post_test and bool(train_cfg.get("run_test_after_train", True)):
        _release_training_gpu_memory(context)
        post_test_context = _with_dependency_context(
            config,
            test_project,
            run_dir=Path(context["run_dir"]),
            overrides=[],
            checkpoint=context.get("report_checkpoint"),
            skip_report=skip_report,
        )
        context["post_test"] = post_test_context
    elif not skip_report and bool(train_cfg.get("run_report_after_train", True)):
        _with_dependency_context(config, run_offline_eval, config=config, context=context)

    if _should_run_post_train_export(train_cfg=train_cfg, skip_export=skip_export):
        context["post_export"] = _with_dependency_context(
            config,
            export_project,
            run_dir=Path(context["run_dir"]),
            overrides=[],
            checkpoint=context.get("report_checkpoint"),
            output_dir=None,
        )
    return context


def _should_run_post_train_export(*, train_cfg: dict[str, Any], skip_export: bool | None) -> bool:
    if skip_export is True:
        return False
    if skip_export is False:
        return True
    return bool(train_cfg.get("run_export_after_train", False))


def _open_snapshot_stage(
    *,
    stage: str,
    run_dir: Path,
    overrides: list[str] | None,
    output_dir: str | Path | None,
    checkpoint: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    config, context = _prepare_snapshot_stage(
        stage=stage,
        run_dir=Path(run_dir).resolve(),
        overrides=overrides or [],
        output_dir=output_dir,
    )
    start_run_descriptor(run_dir=context["run_dir"], config=config)
    load_project_python(Path(context["project_dir"]))
    validate_composed_config(config)
    configure_ram_guard(config)
    _seed_runtime(config)
    _configure_torch_runtime(config)
    selected_checkpoint = _resolve_checkpoint(stage=stage, checkpoint=checkpoint, config=config, context=context)
    return config, context, selected_checkpoint


def _instantiate_stage(config: dict[str, Any], context: dict[str, Any], checkpoint: str | None) -> tuple[Any, Any]:
    datamodule_obj = _with_dependency_context(
        config,
        instantiate,
        _prepare_datamodule_config(config=config, project_dir=Path(context["project_dir"])),
    )
    module_obj = _with_dependency_context(config, instantiate, _prepare_module_config(config=config, checkpoint=checkpoint))
    context["datamodule"] = datamodule_obj
    context["module"] = module_obj
    context["model"] = module_obj
    return datamodule_obj, module_obj


def test_project(
    run_dir: Path,
    overrides: list[str] | None = None,
    checkpoint: str | None = None,
    skip_report: bool = False,
) -> dict[str, Any]:
    config, context, selected_checkpoint = _open_snapshot_stage(
        stage="test", run_dir=run_dir, overrides=overrides, output_dir=None, checkpoint=checkpoint
    )

    _with_dependency_context(config, run_hook_group, config, "pre_build", context)
    datamodule_obj, module_obj = _instantiate_stage(config, context, selected_checkpoint)

    _with_dependency_context(config, run_hook_group, config, "pre_test", context)
    teia_log(
        "test start",
        run_dir=context["run_dir"],
        checkpoint=selected_checkpoint,
    )
    stage_result = _with_dependency_context(
        config,
        _run_stage_module,
        stage="test",
        config=config,
        datamodule_obj=datamodule_obj,
        module_obj=module_obj,
        checkpoint=selected_checkpoint,
        run_dir=Path(context["run_dir"]),
        project_dir=Path(context["project_dir"]),
    )
    context.update(stage_result)
    context["result"] = stage_result["result"]
    context["report_checkpoint"] = selected_checkpoint
    teia_log(
        "test complete",
        run_dir=context["run_dir"],
        checkpoint=selected_checkpoint,
    )
    del stage_result  # drop the extra trainer ref before any fallback prediction/materialization path
    _with_dependency_context(config, run_hook_group, config, "post_test", context)
    if not skip_report and bool((config.get("test") or {}).get("run_report", True)):
        _with_dependency_context(config, run_offline_eval, config=config, context=context)
    _write_stage_metadata(context=context, stage="test")
    finish_run_stage(run_dir=context["run_dir"], stage="test", routes=_module_routes(module_obj))
    return context


def infer_project(
    run_dir: Path,
    overrides: list[str] | None = None,
    checkpoint: str | None = None,
    output_dir: str | Path | None = None,
    src: str | Path | None = None,
) -> dict[str, Any]:
    config, context, selected_checkpoint = _open_snapshot_stage(
        stage="infer", run_dir=run_dir, overrides=overrides, output_dir=output_dir, checkpoint=checkpoint
    )
    datamodule_obj, module_obj = _instantiate_stage(config, context, selected_checkpoint)
    if src is not None:
        if not isinstance(datamodule_obj, InferSourceBindable):
            raise RuntimeError(
                f"--src is not supported by datamodule {type(datamodule_obj).__name__}: "
                "no bind_infer_source hook."
            )
        datamodule_obj.bind_infer_source(src)
    context["report_checkpoint"] = selected_checkpoint
    context["result"] = _with_dependency_context(
        config,
        run_infer,
        config=config,
        context=context,
        datamodule=datamodule_obj,
        module=module_obj,
        checkpoint=selected_checkpoint,
        output_dir=Path(context["infer_dir"]),
    )
    _write_stage_metadata(context=context, stage="infer")
    finish_run_stage(run_dir=context["run_dir"], stage="infer", routes=_module_routes(module_obj))
    return context


def export_project(
    run_dir: Path,
    overrides: list[str] | None = None,
    checkpoint: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config, context, selected_checkpoint = _open_snapshot_stage(
        stage="export", run_dir=run_dir, overrides=overrides, output_dir=output_dir, checkpoint=checkpoint
    )
    datamodule_obj, module_obj = _instantiate_stage(config, context, selected_checkpoint)
    _load_datamodule_artifacts(
        datamodule_obj,
        run_dir=Path(context["run_dir"]),
        required=True,
    )
    context["report_checkpoint"] = selected_checkpoint
    teia_log(
        "export start",
        run_dir=context["run_dir"],
        checkpoint=selected_checkpoint,
        output_dir=context["export_dir"],
    )
    context["result"] = _with_dependency_context(
        config,
        export_bundle,
        model=module_obj,
        config=config,
        run_dir=Path(context["run_dir"]),
        checkpoint=selected_checkpoint,
        output_dir=Path(context["export_dir"]),
        datamodule=datamodule_obj,
    )
    teia_log(
        "export complete",
        run_dir=context["run_dir"],
        bundle=context["result"],
    )
    _write_stage_metadata(context=context, stage="export")
    finish_run_stage(run_dir=context["run_dir"], stage="export", routes=_module_routes(module_obj))
    return context


def _with_dependency_context(composed_config: dict[str, Any], fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise_with_dependency_context(exc, composed_config)


def _run_stage_module(
    *,
    stage: str,
    config: dict[str, Any],
    datamodule_obj: Any,
    module_obj: Any,
    checkpoint: str | None,
    run_dir: Path,
    project_dir: Path,
    restore_weights_only: bool = False,
) -> dict[str, Any]:
    if isinstance(module_obj, NativeStageRunner):
        try:
            result = module_obj.run_teia_stage(
                stage=stage,
                config=config,
                datamodule=datamodule_obj,
                run_dir=run_dir,
                project_dir=project_dir,
                checkpoint=checkpoint,
            )
        except AttributeError:
            pass  # not a native runner after all — fall through to the Lightning path
        else:
            return {"result": result, "callbacks": [], "loggers": [], "trainer": None}

    lightning_result = _run_lightning_stage(
        stage=stage,
        config=config,
        datamodule_obj=datamodule_obj,
        module_obj=module_obj,
        checkpoint=checkpoint,
        run_dir=run_dir,
        restore_weights_only=restore_weights_only,
    )
    return lightning_result


def _drop_fit_only_callbacks(callbacks: list[Any]) -> list[Any]:
    """Drop callbacks that are only meaningful during ``fit`` before a test/predict trainer.

    Lightning's ``WeightAveraging`` only builds its averaged model in the fit stage; outside
    fit it cannot restore that state and merely emits a load-checkpoint warning. The saved
    checkpoint's ``state_dict`` already holds the EMA-averaged weights, so the model under
    test/predict is the EMA model regardless — the callback is a pure no-op here. Excluding
    it keeps the EMA-evaluation behavior and removes the spurious warning.
    """
    return [
        cb
        for cb in callbacks
        if not any(klass.__name__ == "WeightAveraging" for klass in type(cb).__mro__)
    ]


def _ensure_walltime_callback(callbacks: list[Any]) -> list[Any]:
    """Guarantee the wall-time report runs for every Lightning stage across all packs.

    Injected here rather than in each pack's callbacks config so core, vision, tabular, and any
    future pack emit ``eval/walltime.yaml`` by default without per-pack wiring.
    """
    if any(isinstance(cb, WalltimeCallback) for cb in callbacks):
        return callbacks
    return [*callbacks, WalltimeCallback()]


def _run_lightning_stage(
    stage: str,
    config: dict[str, Any],
    datamodule_obj: Any,
    module_obj: Any,
    checkpoint: str | None,
    run_dir: Path,
    restore_weights_only: bool = False,
) -> dict[str, Any]:
    try:
        lightning = __import__("lightning.pytorch", fromlist=["Trainer"])
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Lightning is required to execute Teia modules without a native runner.") from exc
    _suppress_lightning_litlogger_tip()

    try:
        import torch
        register_safe_checkpoint_globals(torch_module=torch)
    except Exception:
        pass

    if (config.get("runtime") or {}).get("channels_last"):
        import torch

        module_obj.to(memory_format=torch.channels_last)

    trainer_kwargs = strip_internal_metadata(config.get("trainer") or {})
    apply_trainer_determinism(config, trainer_kwargs)
    trainer_kwargs.setdefault("default_root_dir", str(run_dir))
    callbacks_cfg = _resolve_runtime_paths(strip_internal_metadata(config.get("callbacks") or {}), run_dir=run_dir)
    loggers_cfg = _resolve_runtime_paths(strip_internal_metadata(config.get("loggers") or {}), run_dir=run_dir)
    callbacks = instantiate_many(callbacks_cfg)
    if stage != "train":
        callbacks = _drop_fit_only_callbacks(callbacks)
    callbacks = _ensure_walltime_callback(callbacks)
    loggers = instantiate_many(loggers_cfg)
    if isinstance(datamodule_obj, TaskMetadataReady):
        datamodule_obj.ensure_task_metadata()
    if stage in {"test", "predict"}:
        _load_datamodule_artifacts(
            datamodule_obj,
            run_dir=run_dir,
            required=True,
        )
    eval_settings, eval_graph = _bind_task_contract(config, datamodule_obj, module_obj)
    if isinstance(module_obj, TeiaStageModule):
        module_obj.configure_from_datamodule(datamodule_obj)
    if stage in {"train", "test"} and eval_graph:
        callbacks = [
            *callbacks,
            CaptureCallback(graph=eval_graph, ctx=eval_settings.ctx, max_rows=eval_settings.max_rows, run_dir=run_dir),
        ]
    if isinstance(module_obj, TeiaStageModule) and isinstance(datamodule_obj, BatchMetaSource):
        module_obj.dry_run(datamodule_obj)
    if stage == "predict":
        # Single device, no DDP: trainer.predict must return outputs in dataloader order so
        # they zip with the source records during materialization.
        trainer_kwargs["devices"] = 1
        trainer_kwargs["num_nodes"] = 1
        trainer_kwargs.pop("strategy", None)
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    if loggers:
        trainer_kwargs["logger"] = loggers if len(loggers) > 1 else loggers[0]
    if "log_every_n_steps" in trainer_kwargs:
        try:
            datamodule_obj.setup("fit")
            n_batches = len(datamodule_obj.train_dataloader())
            if n_batches > 0:
                trainer_kwargs["log_every_n_steps"] = min(trainer_kwargs["log_every_n_steps"], n_batches)
        except Exception:
            pass
    trainer = lightning.Trainer(**trainer_kwargs)
    if stage == "train":
        _hand_manual_optimization_controls(trainer, module_obj)

    try:
        if stage == "train":
            fit_checkpoint = checkpoint or (config.get("train") or {}).get("resume_from")
            if restore_weights_only:
                if checkpoint:
                    _restore_model_weights(module_obj, checkpoint)
                fit_checkpoint = None
            with _suppress_intentional_eval_mode_warning(module_obj), _suppress_model_summary_precision_warning(
                trainer.precision
            ), _suppress_lightning_pytree_deprecation():
                result = trainer.fit(
                    model=module_obj,
                    datamodule=datamodule_obj,
                    ckpt_path=fit_checkpoint,
                    **_trainer_checkpoint_load_kwargs(trainer.fit, fit_checkpoint),
                )
            _save_datamodule_artifacts(datamodule_obj, run_dir=run_dir)
        elif stage == "test":
            test_checkpoint = checkpoint or (config.get("test") or {}).get("checkpoint")
            # Load weights only and test the in-memory module: the saved state_dict already holds the
            # EMA-averaged weights, and passing ckpt_path=None skips Lightning's callback-state restore,
            # which otherwise warns that the dropped fit-only WeightAveraging callback is missing.
            _restore_model_weights(module_obj, test_checkpoint)
            with _suppress_lightning_pytree_deprecation():
                for extra_split in sorted(required_splits(config.get("evalmodule")) - {"test"}):
                    if extra_split != "val":
                        raise ValueError(f"evalmodule reads capture split '{extra_split}', which no stage can capture (only 'val' and 'test').")
                    trainer.validate(model=module_obj, datamodule=datamodule_obj, ckpt_path=None)
                result = trainer.test(
                    model=module_obj,
                    datamodule=datamodule_obj,
                    ckpt_path=None,
                )
        elif stage == "predict":
            predict_checkpoint = checkpoint or (config.get("infer") or {}).get("checkpoint")
            _restore_model_weights(module_obj, predict_checkpoint)
            with _suppress_lightning_pytree_deprecation():
                result = trainer.predict(
                    model=module_obj,
                    datamodule=datamodule_obj,
                    ckpt_path=None,
                    return_predictions=True,
                )
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"Unsupported lightning stage: {stage}")
    except RuntimeError as exc:
        raise_enriched_determinism_error(config, exc)
        raise

    return {
        "result": result,
        "trainer": trainer,
        "callbacks": callbacks,
        "loggers": loggers,
    }


def run_predict_stage(
    *,
    config: dict[str, Any],
    datamodule_obj: Any,
    module_obj: Any,
    checkpoint: str | None,
    run_dir: Path,
) -> dict[str, Any]:
    """Run ``trainer.predict`` over the datamodule's predict split, reusing the shared
    Lightning-stage construction (callbacks, loggers, prediction-ctx, default tqdm bar).

    ``result`` is the ordered list of per-batch routed dicts (keyed by activation-node name).
    """
    return _run_lightning_stage(
        stage="predict",
        config=config,
        datamodule_obj=datamodule_obj,
        module_obj=module_obj,
        checkpoint=checkpoint,
        run_dir=Path(run_dir),
    )


def _hand_manual_optimization_controls(trainer: Any, module_obj: Any) -> None:
    """Stash trainer-level accumulation/clipping on the manual module and clear them on the trainer.

    Lightning's validator forbids ``accumulate_grad_batches != 1`` / ``gradient_clip_val`` under
    manual optimization; the module applies the stashed values itself in ``training_step``.
    """
    if getattr(module_obj, "automatic_optimization", True):
        return
    module_obj._manual_accumulate_grad_batches = int(getattr(trainer, "accumulate_grad_batches", 1) or 1)
    clip_val = getattr(trainer, "gradient_clip_val", None)
    module_obj._manual_grad_clip_val = float(clip_val) if clip_val else None
    module_obj._manual_grad_clip_algorithm = getattr(trainer, "gradient_clip_algorithm", None) or "norm"
    trainer.accumulate_grad_batches = 1
    trainer.gradient_clip_val = None
    trainer.gradient_clip_algorithm = None


def _artifacts_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "artifacts"


def _load_datamodule_artifacts(
    datamodule_obj: Any,
    *,
    run_dir: Path,
    required: bool,
) -> None:
    if isinstance(datamodule_obj, ArtifactPersistable):
        datamodule_obj.load_artifacts(_artifacts_dir(run_dir), required=required)


def _save_datamodule_artifacts(datamodule_obj: Any, *, run_dir: Path) -> None:
    if isinstance(datamodule_obj, ArtifactPersistable):
        datamodule_obj.save_artifacts(_artifacts_dir(run_dir))


def _bind_task_contract(config: dict[str, Any], datamodule_obj: Any, module_obj: Any) -> tuple[Any, dict[str, Any]]:
    """Instantiate ``task._target_``, check boundary A against the planned datamodule, and hand the
    contract plus the evalmodule's ``batch.*`` reads to the module's capture map. Returns the
    evalmodule's ``(settings, graph)``."""
    settings, graph = split_evalmodule(config.get("evalmodule"))
    contract = instantiate(config["task"]) if config.get("task") else None
    if contract is not None and isinstance(datamodule_obj, BatchMetaSource):
        contract.check_data(datamodule_obj.batch_meta(), datamodule_obj.meta(), datamodule_obj.batch_type()._fields)
    module_obj.contract = contract
    records = build_eval_graph(discover_node_entries(graph)) if graph else []
    module_obj.eval_batch_keys = {key for record in records for key in record.in_key if key.startswith("batch.")}
    return settings, graph


# Mixed-precision modes keep fp32 master weights, so ModelSummary's 32-bit size estimate is
# correct, but its lookup table lacks the "*-mixed" keys and warns anyway. Suppress just these.
_KNOWN_GOOD_MIXED_PRECISIONS = {"16-mixed", "bf16-mixed"}

_LIGHTNING_RANK_ZERO_LOGGERS = (
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.utilities.rank_zero",
)
_LITLOGGER_TIP_PREFIX = "💡 Tip: For seamless cloud logging and experiment tracking,"


class _LightningLitLoggerTipFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_LITLOGGER_TIP_PREFIX)


def _suppress_lightning_litlogger_tip() -> None:
    """Suppress Lightning's litlogger promo through its rank-zero logging channel."""
    for logger_name in _LIGHTNING_RANK_ZERO_LOGGERS:
        logger = logging.getLogger(logger_name)
        if any(isinstance(filter_, _LightningLitLoggerTipFilter) for filter_ in logger.filters):
            continue
        logger.addFilter(_LightningLitLoggerTipFilter())


@contextlib.contextmanager
def _suppress_model_summary_precision_warning(precision: Any):
    """Silence the model-summary precision warning for known-good mixed-precision modes only."""
    if str(precision) not in _KNOWN_GOOD_MIXED_PRECISIONS:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Precision .* is not supported by the model summary\..*",
            category=UserWarning,
        )
        yield


@contextlib.contextmanager
def _suppress_lightning_pytree_deprecation():
    """Silence Lightning's internal LeafSpec deprecation; we don't call the deprecated API."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*isinstance\(treespec, LeafSpec\).* is deprecated.*",
        )
        yield


@contextlib.contextmanager
def _suppress_intentional_eval_mode_warning(module_obj: Any):
    if not _allows_eval_modules_at_train_start(module_obj):
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Found \d+ module\(s\) in eval mode at the start of training\..*",
            category=Warning,
        )
        yield


def _allows_eval_modules_at_train_start(module_obj: Any) -> bool:
    # TeiaNetModule exposes this as a plain bool property; non-Teia modules default to False.
    return bool(getattr(module_obj, "allow_eval_modules_at_train_start", False))


def _release_training_gpu_memory(context: dict[str, Any]) -> None:
    """Free GPU memory from the training pass so the next stage has a clean budget."""
    import gc

    try:
        import torch
    except Exception:
        return

    context.pop("trainer", None)
    for key in ("module", "model"):
        obj = context.get(key)
        if obj is not None:
            try:
                obj.cpu()
            except Exception:
                pass
            try:
                obj._trainer = None  # break cyclic ref so Python frees optimizer state
            except Exception:
                pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trainer_checkpoint_load_kwargs(method: Any, checkpoint: str | None) -> dict[str, Any]:
    if not checkpoint:
        return {}
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    if "weights_only" not in parameters:
        return {}
    return {"weights_only": False}


def _restore_model_weights(module_obj: Any, checkpoint: str | None) -> None:
    if not checkpoint:
        return
    try:
        import torch  # pragma: no cover - optional
    except Exception as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError("Torch is required to restore model weights for continued training.") from exc

    payload = load_trusted_checkpoint(checkpoint, map_location="cpu", torch_module=torch)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    loader = getattr(module_obj, "load_state_dict", None)
    if not callable(loader):
        raise TypeError(f"Configured module {type(module_obj)!r} does not support load_state_dict().")
    loader(state_dict, strict=True)


def _prepare_train_context(
    *,
    project_dir: Path,
    config: dict[str, Any],
    hydra_payload: dict[str, Any],
    source_conf_root: Path,
    run_name: str | None,
    overrides: list[str],
) -> dict[str, Any]:
    layout = resolve_runtime_layout(config=config, project_dir=project_dir, cli_run_name=run_name)
    ensure_dir(layout.runs_root)
    ensure_dir(layout.run_dir)
    _apply_runtime_layout_metadata(config, layout)
    return {
        "stage": "train",
        "project_dir": layout.project_dir,
        "run_dir": layout.run_dir,
        "config_dir": layout.config_dir,
        "logs_dir": layout.logs_dir,
        "eval_dir": layout.eval_dir,
        "infer_dir": layout.infer_dir,
        "export_dir": layout.export_dir,
        "artifacts_dir": layout.artifacts_dir,
        "cache_root": layout.cache_root,
        "layout": layout,
        "config": config,
        "overrides": list(overrides),
        "report_cache": {},
        "source_conf_root": source_conf_root,
        "hydra_config": hydra_payload,
    }


def _prepare_snapshot_stage(
    *,
    stage: str,
    run_dir: Path,
    overrides: list[str],
    output_dir: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = compose_run_config(run_dir=run_dir, overrides=overrides)
    saved = load_saved_composed_config(run_dir)
    runtime_meta = saved.get("__runtime_layout__") or {}
    config["__runtime_layout__"] = runtime_meta

    project_dir = Path(str(config.get("__project_dir__") or saved.get("__project_dir__") or run_dir.parent)).resolve()
    cache_root = _resolve_root_value(runtime_meta.get("cache_root") or config.get("cache_root", ".teia-cache"), project_dir=project_dir)
    layout = RuntimeLayout(
        project_dir=project_dir,
        runs_root=_resolve_root_value(runtime_meta.get("runs_root") or config.get("runs_root", "runs"), project_dir=project_dir),
        cache_root=cache_root,
        run_name=str(runtime_meta.get("run_name") or run_dir.name),
        run_dir=run_dir,
        config_dir=run_dir / "config",
        logs_dir=run_dir / "logs",
        eval_dir=run_dir / "eval",
        infer_dir=run_dir / "infer",
        export_dir=run_dir / "export",
        artifacts_dir=run_dir / "artifacts",
    )

    stage_output_dir = None
    if output_dir is not None:
        stage_output_dir = Path(output_dir).resolve()
        ensure_dir(stage_output_dir)

    context = {
        "stage": stage,
        "project_dir": project_dir,
        "run_dir": run_dir,
        "config_dir": layout.config_dir,
        "logs_dir": layout.logs_dir,
        "eval_dir": layout.eval_dir,
        "infer_dir": stage_output_dir or layout.infer_dir,
        "export_dir": stage_output_dir or layout.export_dir,
        "artifacts_dir": layout.artifacts_dir,
        "cache_root": layout.cache_root,
        "layout": layout,
        "config": config,
        "overrides": list(overrides),
        "report_cache": {},
    }
    return config, context


def _persist_train_snapshots(context: dict[str, Any]) -> None:
    config_dir = ensure_dir(Path(context["config_dir"]))
    source_conf_root = Path(context["source_conf_root"])
    snapshot_root = ensure_dir(config_dir / "conf")
    if source_conf_root.exists():
        copy_tree(source_conf_root, snapshot_root)
    write_yaml(config_dir / "composed.yaml", context["config"])
    write_yaml(config_dir / "overrides.yaml", {"overrides": context.get("overrides", [])})
    write_yaml(config_dir / "hydra.yaml", context.get("hydra_config") or {})


def _seed_runtime(config: dict[str, Any]) -> None:
    seed = (config.get("runtime") or {}).get("seed")
    if seed is None:
        return
    random.seed(int(seed))
    try:
        import numpy as np  # pragma: no cover - optional

        np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch  # pragma: no cover - optional

        torch.manual_seed(int(seed))
    except Exception:
        pass


def _configure_torch_runtime(config: dict[str, Any]) -> None:
    runtime_cfg = config.get("runtime") or {}
    matmul_precision = runtime_cfg.get("float32_matmul_precision")
    try:
        import torch  # pragma: no cover - optional
    except Exception:
        return
    setter = getattr(torch, "set_float32_matmul_precision", None)
    if matmul_precision and callable(setter):
        setter(str(matmul_precision))
    configure_torch_determinism(config)


def _module_routes(module_obj: Any) -> dict[str, list[str]]:
    """``{route: [atoms]}`` from the netmodule's ``capture:`` map, for ``artifacts/run.yaml``."""
    routes: dict[str, list[str]] = {}
    for atom in getattr(module_obj, "capture_map_keys", lambda: [])():
        route, name = atom.split(".", 1)
        routes.setdefault(route, []).append(name)
    return routes


def _write_stage_metadata(context: dict[str, Any], stage: str) -> None:
    artifacts_dir = ensure_dir(Path(context["artifacts_dir"]))
    best_checkpoint = _find_checkpoint_path(callbacks=context.get("callbacks") or [], run_dir=Path(context["run_dir"]), kind="best")
    last_checkpoint = _find_checkpoint_path(callbacks=context.get("callbacks") or [], run_dir=Path(context["run_dir"]), kind="last")
    stage_payload = {
        "stage": stage,
        "task": (context["config"].get("task") or {}).get("_target_"),
        "project_dir": str(context["project_dir"]),
        "run_dir": str(context["run_dir"]),
        "report_checkpoint": context.get("report_checkpoint"),
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": last_checkpoint,
        "overrides": list(context.get("overrides", [])),
        "train_mode": context.get("train_mode", "fresh"),
        "parent_run_dir": context.get("parent_run_dir"),
        "parent_checkpoint": context.get("parent_checkpoint"),
        "from_run_weights": context.get("from_run_weights"),
    }
    write_json(artifacts_dir / "metadata.json", stage_payload)
    write_json(artifacts_dir / "stages" / f"{stage}.json", stage_payload)


def _resolve_runtime_paths(payload: Any, run_dir: Path | None) -> Any:
    if run_dir is None:
        return payload
    if isinstance(payload, dict):
        resolved: dict[str, Any] = {}
        for key, value in payload.items():
            if key in {"save_dir", "dirpath"} and isinstance(value, str) and value and not Path(value).is_absolute():
                resolved[key] = str(run_dir / value)
            else:
                resolved[key] = _resolve_runtime_paths(value, run_dir=run_dir)
        return resolved
    if isinstance(payload, list):
        return [_resolve_runtime_paths(item, run_dir=run_dir) for item in payload]
    return payload


def _apply_runtime_layout_metadata(config: dict[str, Any], layout: RuntimeLayout) -> None:
    config["run_name"] = layout.run_name
    config["__runtime_layout__"] = {
        "runs_root": str(layout.runs_root),
        "cache_root": str(layout.cache_root),
        "run_dir": str(layout.run_dir),
        "run_name": layout.run_name,
    }


#: The executors a module preset configures; presets carry no ``_target_`` (teia:core/config.md#modules).
_DATAMODULE_TARGET = "teia.core.datamodule.executor.TeiaDataModule"
_NETMODULE_TARGET = "teia.core.module.net.TeiaNetModule"


def _with_executor_target(module_config: Any, target: str, group: str) -> dict[str, Any]:
    """The default executor unless a (project) module names its own ``_target_`` (a custom Lightning module)."""
    if not isinstance(module_config, dict) or not module_config:
        raise ValueError(f"No `{group}` selected: pick `task=<task>` or `{group}=<task>/<name>`.")
    return {"_target_": target, "_recursive_": False, **module_config}


def _prepare_datamodule_config(config: dict[str, Any], project_dir: Path) -> Any:
    load_project_python(project_dir)
    datamodule_config = _with_executor_target(strip_internal_metadata(config["datamodule"]), _DATAMODULE_TARGET, "datamodule")
    if "root" in datamodule_config:
        raise ValueError("Config datamodule.root is not supported; configure dataset location with top-level data_root.")
    if "data_root" in datamodule_config:
        raise ValueError("Config datamodule.data_root is not supported; configure dataset location with top-level data_root.")
    target = str(datamodule_config.get("_target_", ""))
    if _callable_accepts_kwarg(target, "project_dir"):
        datamodule_config.setdefault("project_dir", str(project_dir))
    data_root = config.get("data_root")
    if data_root not in (None, "???"):
        if not _callable_accepts_kwarg(target, "data_root"):
            raise ValueError(f"Configured datamodule target {target!r} does not accept `data_root`.")
        resolved_data_root = _resolve_root_value(data_root, project_dir=project_dir)
        if not resolved_data_root.exists():
            raise FileNotFoundError(
                f"data_root does not exist: {resolved_data_root}\n"
                f"(configured as {data_root!r} relative to project {project_dir})"
            )
        if resolved_data_root.is_file():
            resolved_data_root = resolved_data_root.parent
        datamodule_config["data_root"] = str(resolved_data_root)
    return datamodule_config


def _prepare_module_config(config: dict[str, Any], checkpoint: str | None) -> Any:
    module_config = _with_executor_target(strip_internal_metadata(config["netmodule"]), _NETMODULE_TARGET, "netmodule")
    module_config["_convert_"] = "all"
    if checkpoint:
        return disable_pretrained_loading(module_config)
    return module_config


def _resolve_root_value(value: Any, project_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (project_dir / path).resolve()


def _callable_accepts_kwarg(target: str, kwarg: str) -> bool:
    try:
        factory = import_string(target)
    except Exception:
        return True

    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return False

    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters) or any(
        parameter.name == kwarg for parameter in parameters
    )


def _resolve_checkpoint(stage: str, checkpoint: str | None, config: dict[str, Any], context: dict[str, Any]) -> str | None:
    if checkpoint:
        return str(Path(checkpoint))

    stage_cfg = config.get(stage) or {}
    configured = stage_cfg.get("checkpoint")
    if configured:
        return str(Path(configured))

    callbacks = context.get("callbacks") or []
    run_dir = Path(context["run_dir"])
    return _find_checkpoint_path(callbacks=callbacks, run_dir=run_dir, kind="best") or _find_checkpoint_path(
        callbacks=callbacks,
        run_dir=run_dir,
        kind="last",
    )


def _prepare_train_bootstrap(
    *,
    project_dir: Path | None,
    config_dir: Path | None,
    overrides: list[str],
    checkpoint: str | None,
    from_run_dir: Path | None,
    from_run_weights: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[str], dict[str, Any], str | None]:
    if from_run_dir is None:
        resolved_project_dir = Path(project_dir or ".").resolve()
        source_conf_root = Path(config_dir).resolve() if config_dir else (resolved_project_dir / "conf")
        config, hydra_payload = compose_overlay_config(
            project_dir=resolved_project_dir,
            config_dir=config_dir,
            overrides=overrides,
        )
        config["__project_dir__"] = str(resolved_project_dir)
        return (
            config,
            hydra_payload,
            resolved_project_dir,
            source_conf_root,
            list(overrides),
            {"train_mode": "fresh", "parent_run_dir": None, "parent_checkpoint": None, "from_run_weights": None},
            checkpoint,
        )

    if config_dir is not None:
        raise ValueError("Continued training does not support --config-dir; use --from-run-dir only.")
    if project_dir is not None:
        raise ValueError("Continued training does not support --project-dir; use --from-run-dir only.")

    resolved_from_run_dir = resolve_resumable_run_dir(from_run_dir)
    config, hydra_payload, saved, saved_overrides, source_conf_root = compose_run_config_tree(
        run_dir=resolved_from_run_dir,
        overrides=overrides,
    )
    project_dir_value = config.get("__project_dir__") or saved.get("__project_dir__") or resolved_from_run_dir.parent
    resolved_project_dir = Path(str(project_dir_value)).resolve()
    config["__project_dir__"] = str(resolved_project_dir)

    _validate_continued_train_compatibility(parent_config=saved, continued_config=config, parent_run_dir=resolved_from_run_dir)
    if (config.get("train") or {}).get("resume_from"):
        raise ValueError(
            "Continued training does not support train.resume_from; use --checkpoint or --from-run-weights instead."
        )

    combined_overrides = [*saved_overrides, *overrides]
    parent_checkpoint = _resolve_saved_run_checkpoint(
        run_dir=resolved_from_run_dir,
        selection=from_run_weights,
        explicit_checkpoint=checkpoint,
    )
    return (
        config,
        hydra_payload,
        resolved_project_dir,
        source_conf_root,
        combined_overrides,
        {
            "train_mode": "continued",
            "parent_run_dir": str(resolved_from_run_dir),
            "parent_checkpoint": parent_checkpoint,
            "from_run_weights": from_run_weights,
        },
        parent_checkpoint,
    )


def _validate_continued_train_compatibility(
    *,
    parent_config: dict[str, Any],
    continued_config: dict[str, Any],
    parent_run_dir: Path,
) -> None:
    for group in ("datamodule", "netmodule"):
        parent_target = (parent_config.get(group) or {}).get("_target_")
        continued_target = (continued_config.get(group) or {}).get("_target_")
        if parent_target != continued_target:
            raise ValueError(
                f"Continued training from {parent_run_dir} cannot change {group}._target_ "
                f"from {parent_target!r} to {continued_target!r}."
            )


def _resolve_saved_run_checkpoint(run_dir: Path, selection: str, explicit_checkpoint: str | None = None) -> str | None:
    if explicit_checkpoint:
        return str(Path(explicit_checkpoint))

    metadata = _load_saved_stage_metadata(run_dir=run_dir, stage="train")
    if selection == "best":
        for key in ("report_checkpoint", "best_checkpoint"):
            candidate = metadata.get(key) if isinstance(metadata, dict) else None
            if candidate:
                return str(Path(str(candidate)))
        return _find_checkpoint_path(callbacks=[], run_dir=run_dir, kind="best") or _find_checkpoint_path(callbacks=[], run_dir=run_dir, kind="last")

    candidate = metadata.get("last_checkpoint") if isinstance(metadata, dict) else None
    if candidate:
        return str(Path(str(candidate)))
    return _find_checkpoint_path(callbacks=[], run_dir=run_dir, kind="last") or _find_checkpoint_path(callbacks=[], run_dir=run_dir, kind="best")


def _load_saved_stage_metadata(run_dir: Path, stage: str) -> dict[str, Any]:
    stage_path = Path(run_dir) / "artifacts" / "stages" / f"{stage}.json"
    if stage_path.exists():
        payload = read_json(stage_path)
        if isinstance(payload, dict):
            return payload

    metadata_path = Path(run_dir) / "artifacts" / "metadata.json"
    if metadata_path.exists():
        payload = read_json(metadata_path)
        if isinstance(payload, dict):
            return payload
    return {}


def _find_checkpoint_path(callbacks: list[Any], run_dir: Path, *, kind: str) -> str | None:
    for callback in callbacks:
        model_path = getattr(callback, f"{kind}_model_path", None)
        if model_path:
            return str(model_path)

    candidates = [run_dir / "weights" / f"{kind}.pt"]
    candidates.extend(_filter_checkpoint_candidates(run_dir, keyword=kind))
    candidates.extend(_filter_checkpoint_candidates(run_dir))
    return _first_existing_path(candidates)


_CHECKPOINT_EXCLUDED_DIRS = {"export", "infer"}


def _filter_checkpoint_candidates(run_dir: Path, keyword: str | None = None) -> list[Path]:
    def _not_excluded(p: Path) -> bool:
        return not any(part in _CHECKPOINT_EXCLUDED_DIRS for part in p.relative_to(run_dir).parts)

    candidates = sorted(p for p in run_dir.rglob("*.ckpt") if _not_excluded(p)) + sorted(
        p for p in run_dir.rglob("*.pt") if _not_excluded(p)
    )
    if keyword is None:
        return candidates
    lowered = keyword.lower()
    return [candidate for candidate in candidates if lowered in candidate.name.lower()]


def _first_existing_path(candidates: list[Path]) -> str | None:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
