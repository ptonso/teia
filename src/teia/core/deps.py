from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import import_module, metadata
from importlib.util import find_spec
from typing import Any

COMPONENT_DISTRIBUTION = "pyteia"
ALL_EXTRA = "all"


class MissingDependencyError(RuntimeError):
    """Raised when a configured Teia run needs optional packages that are not installed."""

    def __init__(
        self,
        message: str,
        *,
        missing_modules: Sequence[str] = (),
        install_commands: Sequence[str] = (),
        notes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.missing_modules = tuple(missing_modules)
        self.install_commands = tuple(install_commands)
        self.notes = tuple(notes)


def require_dependency(
    package_name: str,
    feature: str,
    import_roots: Sequence[str] = (),
) -> None:
    roots = tuple(str(root).split(".", 1)[0] for root in import_roots) or (str(package_name),)
    if any(has_module(root) for root in roots):
        return
    message = render_dependency_message(
        feature=feature,
        missing_modules=(package_name,),
        install_commands=(package_name,),
    )
    raise MissingDependencyError(
        message,
        missing_modules=(package_name,),
        install_commands=(package_name,),
    )


def has_module(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def raise_with_dependency_context(exc: BaseException, config: Any, *, feature: str = "Teia run") -> None:
    enriched = dependency_error_for(exc, config, feature=feature)
    if enriched is not None:
        raise enriched from exc
    raise exc


def dependency_error_for(exc: BaseException, config: Any, *, feature: str = "Teia run") -> MissingDependencyError | None:
    dep = _dependency_cause(exc)
    if dep is None:
        return None

    missing = collect_missing(config)
    notes: list[str] = []
    if isinstance(dep, MissingDependencyError):
        missing.extend(dep.missing_modules)
        notes.extend(dep.notes)
    else:
        name = _missing_module_name(dep)
        if name:
            missing.append(name.split(".", 1)[0])
    missing = _unique(missing)

    matched_extras, direct_packages = _resolve(missing)
    install_commands = _install_commands(matched_extras, direct_packages)
    message = render_dependency_message(
        feature=feature,
        missing_modules=missing,
        install_commands=install_commands,
        notes=notes,
    )
    return MissingDependencyError(
        message,
        missing_modules=missing,
        install_commands=install_commands,
        notes=notes,
    )


def render_dependency_message(
    *,
    feature: str,
    missing_modules: Sequence[str] = (),
    install_commands: Sequence[str] = (),
    groups: Mapping[str, Sequence[str]] | None = None,
    notes: Sequence[str] = (),
) -> str:
    lines = [f"{feature} is missing optional dependencies."]
    mods = _unique(missing_modules)
    if mods:
        lines.append(f"Missing import(s): {', '.join(mods)}.")
    for command in _unique(install_commands):
        lines.append(f"python3 -m pip install {command}")
    if groups:
        listable = {name: tuple(dists) for name, dists in groups.items() if name != ALL_EXTRA}
        if listable:
            lines.append(f"flexible groups (python3 -m pip install {COMPONENT_DISTRIBUTION}[<group>]):")
            for name in sorted(listable):
                lines.append(f"- {name}: {', '.join(listable[name])}")
    for note in _unique(notes):
        lines.append(f"- {note}")
    return "\n".join(lines)


def collect_missing(config: Any) -> list[str]:
    """Run-scoped: trial-import each ``_target_`` module and gather the top-level imports it lacks."""
    missing: list[str] = []
    for module_name in _iter_target_modules(config):
        try:
            import_module(module_name)
        except ModuleNotFoundError as exc:
            name = exc.name or _missing_module_name(exc)
            if name:
                missing.append(name.split(".", 1)[0])
        except Exception:
            continue
    return _unique(missing)


def check_missing(config: Any, *, feature: str) -> None:
    """Proactive dependency gate: raise ``MissingDependencyError`` now if any ``_target_`` under
    ``config`` would fail to import, instead of waiting for the failure at instantiation time.

    Its caller is the eval-graph pre-flight gate (``runtime/validate.py::_validate_eval_deps``),
    a narrowly-scoped exception to this module's otherwise-reactive policy (``dependency_error_for``
    runs only after a real import failure). Everything else about resolution (extras matching,
    install-command rendering) is reused unchanged.
    """
    missing = collect_missing(config)
    if not missing:
        return
    matched_extras, direct_packages = _resolve(missing)
    install_commands = _install_commands(matched_extras, direct_packages)
    message = render_dependency_message(feature=feature, missing_modules=missing, install_commands=install_commands)
    raise MissingDependencyError(message, missing_modules=missing, install_commands=install_commands)


@lru_cache(maxsize=1)
def _component_distributions() -> tuple[str, ...]:
    """Installed ``pyteia`` and every installed distribution that depends on it, when they declare flexible extras."""
    out: list[str] = []
    for dist in metadata.distributions():
        name = _normalize(dist.metadata.get("Name", ""))
        requirements = dist.requires or ()
        depends = any(_normalize(_requirement_name(r)) == COMPONENT_DISTRIBUTION for r in requirements)
        if name != COMPONENT_DISTRIBUTION and not depends:
            continue
        if any(_parse_extra_requirement(requirement)[1] for requirement in requirements):
            out.append(name)
    return tuple(sorted(dict.fromkeys(out)))


@lru_cache(maxsize=1)
def _install_preference() -> tuple[str, ...]:
    others = [name for name in _component_distributions() if name != COMPONENT_DISTRIBUTION]
    return (COMPONENT_DISTRIBUTION, *sorted(others))


@lru_cache(maxsize=1)
def _distribution_extras() -> dict[str, dict[str, tuple[str, ...]]]:
    """``{distribution: {extra: dists}}`` for each installed component distribution.

    The installed component distributions are discovered from metadata at runtime; absent
    distributions are skipped. This is the single source of truth every resolver helper derives from.
    """
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for distribution in _component_distributions():
        try:
            requirements = metadata.requires(distribution) or ()
        except metadata.PackageNotFoundError:
            continue
        groups: dict[str, list[str]] = {}
        for requirement in requirements:
            name, extra = _parse_extra_requirement(requirement)
            if extra and name and _normalize(name) not in {COMPONENT_DISTRIBUTION, distribution}:
                groups.setdefault(extra, []).append(name)
        if groups:
            out[distribution] = {extra: tuple(dict.fromkeys(names)) for extra, names in groups.items()}
    return out


def _extras() -> dict[str, tuple[str, ...]]:
    """The flexible groups unioned across the installed component distributions."""
    merged: dict[str, list[str]] = {}
    for groups in _distribution_extras().values():
        for extra, dists in groups.items():
            merged.setdefault(extra, []).extend(dists)
    return {extra: tuple(dict.fromkeys(dists)) for extra, dists in merged.items()}


def _dist_to_extras() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for extra, dists in _extras().items():
        if extra == ALL_EXTRA:
            continue
        for dist in dists:
            out.setdefault(dist, []).append(extra)
    return out


def _extra_owner(extra: str) -> str:
    """Installable distribution that provides ``extra``, preferring the public ``pyteia``."""
    distribution_extras = _distribution_extras()
    for distribution in _install_preference():
        if extra in distribution_extras.get(distribution, {}):
            return distribution
    return COMPONENT_DISTRIBUTION


def _resolve(missing_roots: Sequence[str]) -> tuple[list[str], list[str]]:
    dist_extras = _dist_to_extras()
    matched: list[str] = []
    direct_packages: list[str] = []
    for root in missing_roots:
        extras = dist_extras.get(_normalize(root))
        if extras:
            matched.extend(extras)
        else:
            direct_packages.append(_normalize(root))
    return _unique(matched), _unique(direct_packages)


def _install_commands(matched_extras: Sequence[str], direct_packages: Sequence[str]) -> list[str]:
    by_owner: dict[str, list[str]] = {}
    for extra in matched_extras:
        by_owner.setdefault(_extra_owner(extra), []).append(extra)
    tokens: list[str] = []
    install_preference = _install_preference()
    preferred = [owner for owner in install_preference if owner in by_owner]
    others = sorted(owner for owner in by_owner if owner not in install_preference)
    for owner in [*preferred, *others]:
        tokens.append(f"{owner}[{','.join(sorted(by_owner[owner]))}]")
    tokens.extend(_unique(direct_packages))
    return [" ".join(tokens)] if tokens else []


_EXTRA_MARKER = re.compile(r"extra\s*==\s*[\"']([^\"']+)[\"']")


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[ @]", requirement.strip(), maxsplit=1)[0].strip().lower()


def _parse_extra_requirement(requirement: str) -> tuple[str, str | None]:
    marker = _EXTRA_MARKER.search(requirement)
    if marker is None:
        return "", None
    return _requirement_name(requirement), marker.group(1)


def _normalize(root: str) -> str:
    return str(root).strip().lower().replace("_", "-")


def _iter_target_modules(config: Any) -> list[str]:
    return _unique(_target_module(target) for target in _iter_targets(config))


def _iter_targets(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        found = [str(value["_target_"])] if value.get("_target_") else []
        for child in value.values():
            found.extend(_iter_targets(child))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        found: list[str] = []
        for child in value:
            found.extend(_iter_targets(child))
        return found
    return []


def _target_module(target: str) -> str:
    module_name, _, _ = str(target).rpartition(".")
    return module_name


def _dependency_cause(exc: BaseException) -> BaseException | None:
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (MissingDependencyError, ModuleNotFoundError, ImportError)):
            return current
        for child in (getattr(current, "__cause__", None), getattr(current, "__context__", None)):
            if isinstance(child, BaseException):
                stack.append(child)
    return None


def _missing_module_name(exc: BaseException) -> str | None:
    name = getattr(exc, "name", None)
    if name:
        return str(name)
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
    return match.group(1) if match else None


def _unique(values: Sequence[str] | Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
