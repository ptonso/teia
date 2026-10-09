"""Bounded source-collection for standalone export bundles.

The export bundle must run without ``teia`` installed (package spec core/export.md). Rather than
shipping one generic template that branches at deploy time, the exporter *collects* the
exact pure-Python source a given model needs — a preprocess kernel and the activation-owned
decode kernel — and emits it verbatim into the bundle.

``collect_callable(fn)`` walks ``fn``'s source, resolves every free name against
``fn.__globals__`` and inlines the transitive closure of allowed helpers, emitting
``import`` lines for allowlisted external/stdlib modules. Anything else (a teia-core
symbol, instance state, a decorator, a closure) raises ``ExportCollectError`` — the
export fails fast naming the offending symbol instead of emitting a broken bundle.

A symbol is collectable when it is a module-level function/constant living in the
caller's own module or in a designated ``*.export.kernels`` package, and references only
those plus allowlisted libraries (numpy, torch, torchvision, math, PIL, stdlib).
"""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from types import ModuleType
from typing import Any, Callable

# Top-level external packages a collected kernel may import.
ALLOWED_EXTERNAL = frozenset({"numpy", "torch", "torchvision", "math", "PIL"})

# Module prefixes whose functions/constants may be inlined verbatim into a bundle. A ``kernel``
# staticmethod that *delegates* to a shared helper in one of these (rather than being that
# helper itself) still collects: the helper is a dependency, not the entry, so it must clear this
# allowlist rather than the free "same module as its caller" rule an entry gets. Other distributions
# add curated modules through the ``teia.export_kernels`` entry-point group.
KERNEL_PACKAGES = (
    "teia.core.export.kernels",
    "teia.node.net._activation_utils",
)
KERNEL_ENTRY_GROUP = "teia.export_kernels"

_MISSING = object()


class ExportCollectError(RuntimeError):
    """Raised when a callable cannot be collected into a standalone bundle."""


@dataclass(slots=True)
class EmittedSource:
    """Result of collecting a callable: import lines + ordered source blocks."""

    entry: str
    imports: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)

    def render(self) -> str:
        parts = ["from __future__ import annotations", ""]
        if self.imports:
            parts.extend(self.imports)
            parts.append("")
        parts.append("\n\n\n".join(self.blocks))
        return "\n".join(parts).rstrip() + "\n"


def collect_callable(fn: Callable[..., Any], owner: str | None = None) -> EmittedSource:
    """Collect ``fn`` and its allowed transitive dependencies into standalone source.

    ``owner``, when given, names the class ``fn`` was read off (e.g. ``type(node).__name__``
    for a ``kernel`` staticmethod). Every concrete class names its kernel method the same way,
    so an entry point literally named ``kernel`` is emitted as ``<owner.lower()>_kernel`` to
    avoid collisions across a multi-node chain; any other name (a shared helper a kernel
    delegates to) is kept as-is.
    """
    fn = inspect.unwrap(fn)
    if not inspect.isfunction(fn):
        raise ExportCollectError(f"Cannot collect {fn!r}: only module-level functions are supported.")

    imports: dict[str, None] = {}
    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    entry_name = fn.__name__

    def process(obj: Any, *, entry: bool = False) -> None:
        nonlocal entry_name
        if isinstance(obj, _Constant):
            key = ("<const>", obj.name)
        else:
            key = (str(getattr(obj, "__module__", "")), str(getattr(obj, "__qualname__", obj)))
        if key in seen:
            return
        seen.add(key)
        if isinstance(obj, _Constant):
            blocks.append(f"{obj.name} = {obj.value!r}")
            return
        source = _read_source(obj)
        tree = ast.parse(source)
        has_staticmethod = _reject_decorators(obj, tree, allow_staticmethod=entry)
        deps: list[Any] = []
        for name in _referenced_names(tree):
            target = obj.__globals__.get(name, _MISSING)
            if target is _MISSING:
                continue
            statement, dependency = _classify(obj, name, target)
            if statement is not None:
                imports.setdefault(statement, None)
            if dependency is not None:
                deps.append(dependency)
        for dependency in deps:
            process(dependency)
        body = source.strip("\n")
        if has_staticmethod:
            body = _strip_staticmethod_decorator(body)
        if entry and owner is not None and obj.__name__ == "kernel":
            new_name = f"{owner.lower()}_{obj.__name__}"
            body = _rename_def(body, obj.__name__, new_name)
            entry_name = new_name
        blocks.append(body)

    process(fn, entry=True)
    return EmittedSource(entry=entry_name, imports=list(imports), blocks=blocks)


def _strip_staticmethod_decorator(source: str) -> str:
    return "\n".join(line for line in source.split("\n") if line.strip() != "@staticmethod")


def _rename_def(source: str, old_name: str, new_name: str) -> str:
    return source.replace(f"def {old_name}(", f"def {new_name}(", 1)


def _read_source(obj: Any) -> str:
    try:
        return textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError) as exc:
        raise ExportCollectError(f"Cannot read source for {obj!r}: {exc}") from exc


def _reject_decorators(obj: Any, tree: ast.Module, *, allow_staticmethod: bool = False) -> bool:
    """Reject any decorator except a lone ``@staticmethod`` on an ``entry`` def, which is how a
    ``kernel`` method is read straight off its class. Returns whether such a
    decorator was found (so the caller can strip it from the emitted source)."""
    has_staticmethod = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.decorator_list:
            if (
                allow_staticmethod
                and len(node.decorator_list) == 1
                and isinstance(node.decorator_list[0], ast.Name)
                and node.decorator_list[0].id == "staticmethod"
            ):
                has_staticmethod = True
                continue
            raise ExportCollectError(
                f"{getattr(obj, '__qualname__', obj)!r} uses decorators; export kernels must be plain functions."
            )
    return has_staticmethod


def _classify(owner: Any, name: str, target: Any) -> tuple[str | None, Any]:
    """Return ``(import_statement, inline_target)`` for a referenced global symbol."""
    if isinstance(target, ModuleType):
        return _module_import(name, target.__name__), None
    if inspect.isfunction(target) or inspect.isclass(target):
        module = str(getattr(target, "__module__", ""))
        if _is_kernel_module(module) or module == owner.__module__:
            return None, target
        if _is_external_module(module):
            return f"from {module} import {name}", None
        raise ExportCollectError(
            f"{owner.__qualname__!r} references {name!r} from {module!r}, which is not a kernel or "
            "allowlisted module. Move the helper into an export kernel package or inline it."
        )
    if isinstance(target, (int, float, str, bool, type(None), list, tuple, dict)):
        return None, _Constant(name, target)
    raise ExportCollectError(
        f"{owner.__qualname__!r} references {name!r} ({type(target).__name__}), which is not export-portable."
    )


@dataclass(slots=True)
class _Constant:
    name: str
    value: Any


def _module_import(binding: str, module_name: str) -> str:
    top = module_name.split(".")[0]
    if top not in ALLOWED_EXTERNAL and top not in sys.stdlib_module_names:
        raise ExportCollectError(f"Kernel imports {module_name!r}, which is not allowed in a standalone bundle.")
    if "." in module_name and module_name.rsplit(".", 1)[1] == binding:
        return f"from {module_name.rsplit('.', 1)[0]} import {binding}"
    if binding == module_name:
        return f"import {module_name}"
    return f"import {module_name} as {binding}"


@lru_cache(maxsize=1)
def _kernel_packages() -> tuple[str, ...]:
    return (*KERNEL_PACKAGES, *(ep.value for ep in metadata.entry_points(group=KERNEL_ENTRY_GROUP)))


def _is_kernel_module(module: str) -> bool:
    return any(module == pkg or module.startswith(pkg + ".") for pkg in _kernel_packages())


def _is_external_module(module: str) -> bool:
    top = module.split(".")[0]
    return top in ALLOWED_EXTERNAL or top in sys.stdlib_module_names


def _referenced_names(tree: ast.AST) -> list[str]:
    collector = _RefCollector()
    collector.visit(tree)
    return list(dict.fromkeys(collector.names))


class _RefCollector(ast.NodeVisitor):
    """Collect Load-context free names, ignoring annotations (stringized by __future__)."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for default in [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]:
            self.visit(default)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_arg(self, node: ast.arg) -> None:  # skip parameter annotations
        return None

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root: ast.AST = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name):
            self.names.append(root.id)
        else:
            self.visit(root)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.append(node.id)
