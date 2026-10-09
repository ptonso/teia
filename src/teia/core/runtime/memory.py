from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


BYTES_PER_GB = 1024**3
BYTES_PER_KB = 1024
MEMINFO_PATH = Path("/proc/meminfo")
STATUS_PATH = Path("/proc/self/status")


@dataclass(frozen=True, slots=True)
class MemoryResources:
    total_bytes: int
    available_bytes: int


@dataclass(slots=True)
class RamGuardMonitor:
    """Watches process RSS and fails fast when it crosses the RAM limit.

    A hard cap is not possible: ``RLIMIT_AS`` caps virtual address space, which
    CUDA reserves tens of GiB of and would break; ``RLIMIT_RSS`` is a kernel
    no-op. So this guard polls resident physical RAM (``VmRSS``), warns as it
    approaches the limit, and hard-exits once the limit is crossed.
    """

    limit_bytes: int
    warn_fraction: float
    interval_seconds: float = 10.0
    read_current_bytes: Callable[[], int] = lambda: read_process_rss_bytes()
    emit: Callable[[str], None] = lambda message: print(message, flush=True)
    abort: Callable[[], None] = lambda: os._exit(137)
    last_warning_at: float | None = None

    def check(self, now: float | None = None) -> bool:
        current = self.read_current_bytes()
        if current >= self.limit_bytes:
            self.emit(
                "[teia] RAM guard tripped: "
                f"rss={_format_gib(current)} limit={_format_gib(self.limit_bytes)} — exiting"
            )
            self.abort()
            return True
        threshold = int(self.limit_bytes * self.warn_fraction)
        if current < threshold:
            return False
        timestamp = time.monotonic() if now is None else float(now)
        if self.last_warning_at is not None and timestamp - self.last_warning_at < self.interval_seconds:
            return False
        self.last_warning_at = timestamp
        self.emit(
            "[teia] RAM guard warning: "
            f"rss={_format_gib(current)} "
            f"threshold={_format_gib(threshold)} "
            f"limit={_format_gib(self.limit_bytes)}"
        )
        return True


_ACTIVE_MONITOR: RamGuardMonitor | None = None
_MONITOR_THREAD: threading.Thread | None = None


def configure_ram_guard(config: dict[str, Any]) -> int | None:
    guard_cfg = (config.get("runtime") or {}).get("ram_guard") or {}
    resources = None if str(guard_cfg.get("type")) == "none" else read_memory_resources()
    limit = resolve_ram_guard_bytes(guard_cfg, resources)
    if limit is None:
        return None
    start_ram_guard(limit, _fraction(guard_cfg, "warn_fraction"))
    return limit


def read_memory_resources(path: Path = MEMINFO_PATH) -> MemoryResources:
    values = _read_kb_fields(path, {"MemTotal", "MemAvailable"})
    total = values["MemTotal"]
    available = values["MemAvailable"]
    if total <= 0 or available <= 0 or available > total:
        raise ValueError(f"Invalid RAM resources: MemTotal={total} MemAvailable={available}")
    return MemoryResources(total_bytes=total, available_bytes=available)


def resolve_ram_guard_bytes(guard_cfg: dict[str, Any], resources: MemoryResources | None) -> int | None:
    kind = str(guard_cfg.get("type"))
    if kind == "none":
        return None
    if kind == "gb":
        return int(_positive_float(guard_cfg, "gb") * BYTES_PER_GB)
    if resources is None:
        raise ValueError("RAM resources are required for percent-based RAM guards.")
    if kind == "total_percent":
        return int(resources.total_bytes * _fraction(guard_cfg, "percent"))
    if kind == "usable_percent":
        return int(resources.available_bytes * _fraction(guard_cfg, "percent"))
    raise ValueError(f"Invalid runtime.ram_guard.type: {kind!r}")


def start_ram_guard(limit_bytes: int, warn_fraction: float) -> RamGuardMonitor:
    global _ACTIVE_MONITOR, _MONITOR_THREAD

    if limit_bytes <= 0:
        raise ValueError(f"runtime.ram_guard resolved to non-positive bytes: {limit_bytes}")
    _ACTIVE_MONITOR = RamGuardMonitor(limit_bytes=limit_bytes, warn_fraction=warn_fraction)
    if _MONITOR_THREAD is None or not _MONITOR_THREAD.is_alive():
        _MONITOR_THREAD = threading.Thread(target=_ram_guard_loop, name="teia-ram-guard", daemon=True)
        _MONITOR_THREAD.start()
    return _ACTIVE_MONITOR


def read_process_rss_bytes(path: Path = STATUS_PATH) -> int:
    return _read_kb_fields(path, {"VmRSS"})["VmRSS"]


def _ram_guard_loop() -> None:
    while True:
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.check()
        time.sleep(1.0)


def _read_kb_fields(path: Path, keys: set[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition(":")
        if key not in keys:
            continue
        parts = raw.split()
        if len(parts) != 2 or parts[1] != "kB":
            raise ValueError(f"Expected {key} in kB in {path}.")
        values[key] = int(parts[0]) * BYTES_PER_KB
    missing = keys - values.keys()
    if missing:
        raise ValueError(f"Missing {', '.join(sorted(missing))} in {path}.")
    return values


def _positive_float(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if value is None:
        raise ValueError(f"runtime.ram_guard.{key} is required.")
    number = float(value)
    if number <= 0:
        raise ValueError(f"runtime.ram_guard.{key} must be > 0.")
    return number


def _fraction(cfg: dict[str, Any], key: str) -> float:
    number = _positive_float(cfg, key)
    if number > 1:
        raise ValueError(f"runtime.ram_guard.{key} must be <= 1.")
    return number


def _format_gib(value: int) -> str:
    return f"{value / BYTES_PER_GB:.2f}GiB"
