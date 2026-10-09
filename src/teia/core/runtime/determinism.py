from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any


DETERMINISM_MODES = ("force", "prefer", "stochastic")
_ORIGINAL_SHOWWARNING: Any | None = None
_SEEN_WARN_ONLY_OPS: set[str] = set()


@dataclass(frozen=True)
class DeterminismPolicy:
    mode: str
    torch_enabled: bool
    torch_warn_only: bool
    lightning: bool | str


def determinism_policy(config: dict[str, Any]) -> DeterminismPolicy:
    mode = determinism_mode(config)
    if mode == "force":
        return DeterminismPolicy(mode, True, False, True)
    if mode == "prefer":
        return DeterminismPolicy(mode, True, True, "warn")
    return DeterminismPolicy(mode, False, False, False)


def determinism_mode(config: dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime") or {}
    determinism_cfg = runtime_cfg.get("determinism", {"mode": "prefer"})
    if not isinstance(determinism_cfg, dict):
        raise ValueError("runtime.determinism must be a mapping with mode: force | prefer | stochastic.")
    mode = determinism_cfg.get("mode", "prefer")
    if mode not in DETERMINISM_MODES:
        expected = ", ".join(DETERMINISM_MODES)
        raise ValueError(f"Invalid runtime.determinism.mode: {mode!r}. Expected one of: {expected}")
    return str(mode)


def configure_torch_determinism(config: dict[str, Any]) -> None:
    policy = determinism_policy(config)
    _configure_determinism_warning_bridge(policy)
    try:
        import torch  # pragma: no cover - optional
    except Exception:
        return
    if policy.torch_enabled:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    elif policy.mode == "stochastic":
        warnings.warn(
            "runtime.determinism.mode=stochastic disables deterministic algorithm enforcement.",
            UserWarning,
            stacklevel=2,
        )
    torch.use_deterministic_algorithms(policy.torch_enabled, warn_only=policy.torch_warn_only)


def _configure_determinism_warning_bridge(policy: DeterminismPolicy) -> None:
    if policy.mode == "prefer":
        _install_determinism_warning_bridge()
    else:
        _restore_determinism_warning_bridge()


def _install_determinism_warning_bridge() -> None:
    global _ORIGINAL_SHOWWARNING
    if warnings.showwarning is _show_teia_determinism_warning:
        return
    if _ORIGINAL_SHOWWARNING is None:
        _ORIGINAL_SHOWWARNING = warnings.showwarning
    warnings.showwarning = _show_teia_determinism_warning


def _restore_determinism_warning_bridge() -> None:
    global _ORIGINAL_SHOWWARNING
    if _ORIGINAL_SHOWWARNING is None:
        return
    if warnings.showwarning is _show_teia_determinism_warning:
        warnings.showwarning = _ORIGINAL_SHOWWARNING
    _ORIGINAL_SHOWWARNING = None


def _show_teia_determinism_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any | None = None,
    line: str | None = None,
) -> None:
    if issubclass(category, UserWarning) and _is_torch_warn_only_determinism_warning(message):
        op = _determinism_warning_operation(message)
        if op not in _SEEN_WARN_ONLY_OPS:
            _SEEN_WARN_ONLY_OPS.add(op)
            print(
                "[teia] determinism warning: "
                f"{op} has no deterministic PyTorch implementation; continuing because "
                "runtime.determinism.mode=prefer. Use runtime.determinism.mode=force to fail fast "
                "or runtime.determinism.mode=stochastic to disable deterministic enforcement.",
                file=file or sys.stderr,
                flush=True,
            )
        return
    if _ORIGINAL_SHOWWARNING is not None:
        _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file, line)
        return
    target = file or sys.stderr
    target.write(warnings.formatwarning(message, category, filename, lineno, line))


def _is_torch_warn_only_determinism_warning(message: Warning | str) -> bool:
    text = str(message)
    return (
        "does not have a deterministic implementation" in text
        and "torch.use_deterministic_algorithms(True, warn_only=True)" in text
    )


def _determinism_warning_operation(message: Warning | str) -> str:
    text = str(message).strip()
    marker = " does not have a deterministic implementation"
    if marker not in text:
        return "A PyTorch operation"
    return text.split(marker, 1)[0].strip() or "A PyTorch operation"


def apply_trainer_determinism(config: dict[str, Any], trainer_kwargs: dict[str, Any]) -> dict[str, Any]:
    policy = determinism_policy(config)
    if "deterministic" in trainer_kwargs:
        raise ValueError(
            "trainer.deterministic is not supported; use runtime.determinism.mode=force|prefer|stochastic."
        )
    if bool(trainer_kwargs.get("benchmark")) and policy.mode in {"force", "prefer"}:
        raise ValueError(f"trainer.benchmark=true conflicts with runtime.determinism.mode={policy.mode}.")
    trainer_kwargs["deterministic"] = policy.lightning
    if policy.torch_enabled:
        trainer_kwargs.setdefault("benchmark", False)
    return trainer_kwargs


def raise_enriched_determinism_error(config: dict[str, Any], exc: RuntimeError) -> None:
    if determinism_mode(config) != "force":
        return
    message = str(exc)
    if "deterministic" not in message and "nondeterministic" not in message:
        return
    raise RuntimeError(
        "Teia runtime.determinism.mode=force failed because PyTorch required a nondeterministic operation.\n"
        f"Original error: {message}\n"
        "To continue with warnings, set runtime.determinism.mode=prefer.\n"
        "To allow nondeterministic kernels, set runtime.determinism.mode=stochastic."
    ) from exc
