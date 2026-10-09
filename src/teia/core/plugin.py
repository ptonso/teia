"""Plugin conformance checks: ``teia plugin check`` (specs/plugin.md).

Static rules read the plugin's source tree. Environment rules need the plugin installed in the running interpreter and
compare it with ``teia`` and every other installed provider of components and configs.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import pkgutil
import re
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

#: The distribution that installs the ``teia`` package.
DISTRIBUTION = "pyteia"
MODULE_GROUPS = ("datamodule", "netmodule", "evalmodule")
_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_TITLE = re.compile(r'^\s*-\s*title:\s*"?(.+?)"?\s*$', re.M)


@dataclass(frozen=True, slots=True)
class Finding:
    level: str
    rule: str
    path: Path
    message: str

    def __str__(self) -> str:
        return f"{self.level:<7} [{self.rule}] {self.path}: {self.message}"


@dataclass(frozen=True, slots=True)
class Plugin:
    root: Path
    name: str
    package: str
    pyproject: dict[str, Any]

    @property
    def src(self) -> Path:
        return self.root / "src" / self.package

    @property
    def conf(self) -> Path:
        return self.src / "conf"


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_plugin(root: Path, namespace: str | None = None) -> Plugin:
    root = root.resolve()
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise SystemExit(f"{root} has no pyproject.toml; a plugin is an installable package.")
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    name = pyproject.get("project", {}).get("name")
    if not name:
        raise SystemExit(f"{pyproject_path} has no [project] name.")
    return Plugin(root=root, name=name, package=namespace or name.replace("-", "_"), pyproject=pyproject)


# -- static rules ------------------------------------------------------------------------------------------------------


def check_pyproject(plugin: Plugin) -> list[Finding]:
    path = plugin.root / "pyproject.toml"
    project = plugin.pyproject.get("project", {})
    problems = []
    if "build-system" not in plugin.pyproject:
        problems.append("no [build-system]")
    if "version" not in project and "version" not in project.get("dynamic", []):
        problems.append("no version")
    if "license" not in project and "license-files" not in project:
        problems.append("no license")
    if plugin.package != "teia" and DISTRIBUTION not in {_norm(dep) for dep in _requirement_names(project.get("dependencies", []))}:
        problems.append(f"does not depend on {DISTRIBUTION}")
    return [Finding("error", "pyproject", path, problem) for problem in problems]


def check_license_file(plugin: Plugin, *, publish: bool) -> list[Finding]:
    if any(plugin.root.glob("LICENSE*")) or any(plugin.root.glob("COPYING*")):
        return []
    return [Finding("error" if publish else "warning", "license-file", plugin.root, "no LICENSE file")]


def check_package(plugin: Plugin) -> list[Finding]:
    src = plugin.root / "src"
    if not plugin.src.is_dir():
        return [Finding("error", "package", plugin.src, f"the package must live in src/{plugin.package}/")]
    others = sorted(p.name for p in src.iterdir() if p.is_dir() and p.name != plugin.package and any(p.rglob("*.py")))
    return [Finding("error", "package", src / other, f"only src/{plugin.package}/ may hold Python") for other in others]


def check_searchpath(plugin: Plugin) -> list[Finding]:
    hydra = plugin.root / "hydra_plugins"
    findings = []
    plugin_file = hydra / f"{plugin.name.replace('-', '_')}_searchpath.py"
    if plugin.conf.is_dir() and not plugin_file.is_file():
        findings.append(Finding("error", "searchpath", plugin_file, "missing: the conf/ tree is never registered"))
    if (hydra / "__init__.py").exists():
        findings.append(Finding("error", "searchpath", hydra / "__init__.py", "hydra_plugins/ must stay a namespace package"))
    setuptools = plugin.pyproject.get("tool", {}).get("setuptools")
    if setuptools is None:
        return findings + [Finding("warning", "searchpath", plugin.root / "pyproject.toml", "not setuptools: cannot verify that hydra_plugins and conf yaml are packaged")]
    include = setuptools.get("packages", {}).get("find", {}).get("include", [])
    if not any(pattern.startswith("hydra_plugins") for pattern in include):
        findings.append(Finding("error", "searchpath", plugin.root / "pyproject.toml", "[tool.setuptools.packages.find] include must list hydra_plugins*"))
    data = setuptools.get("package-data", {})
    if not any("yaml" in pattern for patterns in data.values() for pattern in patterns):
        findings.append(Finding("error", "searchpath", plugin.root / "pyproject.toml", "[tool.setuptools.package-data] must ship the conf *.yaml"))
    return findings


def owner_index(group: str) -> int:
    return 3 if group == "node" else 2 if group in MODULE_GROUPS else 1


def check_conf_owner(plugin: Plugin) -> list[Finding]:
    findings = []
    for path in conf_files(plugin.conf):
        parts = path.relative_to(plugin.conf).parts
        index = owner_index(parts[0])
        if len(parts) <= index + 1 or parts[index] != plugin.package:
            expected = "/".join([*parts[:index], plugin.package, "..."])
            findings.append(Finding("error", "conf-owner", path, f"must live under {expected}"))
    return findings


def check_targets(plugin: Plugin, *, shared: bool = False) -> list[Finding]:
    """In the shared ``teia`` namespace a module missing from the tree may come from ``teia`` itself (target-import decides)."""
    findings = []
    for path, target in conf_targets(plugin.conf):
        if not target.startswith(plugin.package + "."):
            continue
        module, _, symbol = target.rpartition(".")
        source = module_source(plugin, module)
        if source is None and shared:
            continue
        if source is None:
            findings.append(Finding("error", "target", path, f"{target}: module {module} does not exist"))
        elif symbol not in defined_names(source):
            findings.append(Finding("error", "target", path, f"{target}: {symbol} is not defined in {source.relative_to(plugin.root)}"))
    return findings


def check_docstrings(plugin: Plugin) -> list[Finding]:
    findings = []
    for path in component_modules(plugin):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if not tree.body:
            continue
        doc = ast.get_docstring(tree) or ""
        problem = provenance_problem(doc)
        if problem:
            findings.append(Finding("error", "docstring", path, problem))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_") and ast.get_docstring(node) is None:
                findings.append(Finding("error", "docstring", path, f"public {node.name} (line {node.lineno}) has no docstring"))
    return findings


def provenance_problem(doc: str) -> str | None:
    """The first violation of the provenance template (specs/docstrings.md), or ``None``."""
    lines = doc.splitlines()
    if not any(line.startswith("Source:") for line in lines) or "Description:" not in lines:
        return "module docstring needs Source: and Description: (specs/docstrings.md)"
    if "Source: common knowledge" not in lines:
        start = lines.index("Source:") + 1
        end = next((i for i in range(start, len(lines)) if lines[i] and not lines[i].startswith(" ")), len(lines))
        entries = [line for line in lines[start:end] if line.strip()]
        if not entries or len(entries) % 3 or not all(
            title.startswith("  - title: ") and url.startswith("    url: ") and year.startswith("    year: ")
            for title, url, year in zip(entries[::3], entries[1::3], entries[2::3])
        ):
            return "Source: must be `common knowledge` or a list of title/url/year entries"
    if "Upstream:" in lines:
        start = lines.index("Upstream:") + 1
        block = " ".join(line for line in lines[start : start + 3])
        missing = [key for key in ("repo:", "commit:", "license:") if key not in block]
        if missing:
            return f"Upstream: is missing {', '.join(missing)}"
    return None


# -- environment rules -------------------------------------------------------------------------------------------------


def check_installed(plugin: Plugin) -> list[Finding]:
    try:
        dist = metadata.distribution(plugin.name)
    except metadata.PackageNotFoundError:
        return [Finding("error", "installed", plugin.root, f"not installed: pip install -e {plugin.root}")]
    root = editable_root(dist)
    if root != plugin.root:
        return [Finding("error", "installed", plugin.root, f"{plugin.name} is installed from {root or dist.locate_file('')}; reinstall with pip install -e {plugin.root}")]
    return []


def check_registered(plugin: Plugin, layers: list[tuple[str, Path]]) -> list[Finding]:
    if not plugin.conf.is_dir() or plugin.conf.resolve() in {path for _, path in layers}:
        return []
    return [Finding("error", "registered", plugin.conf, "not on the Hydra search path; check hydra_plugins/ and reinstall")]


def check_imports(plugin: Plugin) -> list[Finding]:
    declared = {_norm(name) for name in declared_distributions(plugin)} | {DISTRIBUTION} | {_norm(name) for name in teia_requirements()}
    provided = metadata.packages_distributions()
    findings = []
    for path in python_files(plugin):
        for top in sorted(imported_tops(path.read_text(encoding="utf-8"))):
            if top in sys.stdlib_module_names or top in {plugin.package, "teia", "hydra_plugins", "__future__"}:
                continue
            dists = {_norm(name) for name in provided.get(top, [])}
            if not dists and _norm(top) in declared:
                continue
            if not dists:
                findings.append(Finding("error", "import", path, f"imports {top}, which no installed distribution provides"))
            elif not dists & declared:
                findings.append(Finding("error", "import", path, f"imports {top} from {', '.join(sorted(dists))}, which pyproject does not declare"))
    return findings


def check_target_imports(plugin: Plugin) -> list[Finding]:
    findings = []
    for path, target in conf_targets(plugin.conf):
        try:
            resolve_target(target)
        except Exception as exc:  # noqa: BLE001 - every failure is a finding
            findings.append(Finding("error", "target-import", path, f"{target}: {type(exc).__name__}: {exc}"))
    return findings


def check_base_classes(plugin: Plugin) -> list[Finding]:
    """A node leaf targets a class built on a class the ``teia`` distribution ships: a base class or an extension point."""
    teia_root = _teia_root()
    findings = []
    for path in conf_files(plugin.conf / "node"):
        parts = path.relative_to(plugin.conf).parts
        if parts[1:3] == ("eval", "kernel"):
            continue
        target = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("_target_")
        if not target:
            continue
        try:
            cls = resolve_target(target)
        except Exception:  # noqa: BLE001 - reported by target-import
            continue
        ancestors = cls.__mro__[1:] if isinstance(cls, type) else ()
        if not any((origin := module_origin(base.__module__)) is not None and origin.is_relative_to(teia_root) for base in ancestors):
            findings.append(Finding("error", "base-class", path, f"{target} must subclass a teia base class (teia.base) or a teia extension point"))
    return findings


def check_duplicates(plugin: Plugin, layers: list[tuple[str, Path]]) -> list[Finding]:
    own = plugin.conf.resolve()
    others = [(label, path) for label, path in layers if path != own]
    other_files = {path.relative_to(root).as_posix(): label for label, root in others for path in conf_files(root)}
    findings = [
        Finding("error", "conf-duplicate", path, f"{path.relative_to(plugin.conf).as_posix()} also exists in {other_files[path.relative_to(plugin.conf).as_posix()]}")
        for path in conf_files(plugin.conf)
        if path.relative_to(plugin.conf).as_posix() in other_files
    ]

    own_tasks = {p.stem.replace("_", "-"): p for p in (plugin.src / "task").glob("*.py") if not p.name.startswith("_")}
    other_tasks = contract_tasks(exclude=plugin.src / "task") | {p.stem for _, root in others for p in conf_files(root / "task")}
    findings += [Finding("error", "task-duplicate", path, f"task {name} already exists in another provider") for name, path in own_tasks.items() if name in other_tasks]

    other_targets = [target for _, root in others for _, target in conf_targets(root / "node")]
    other_classes = {target.rsplit(".", 1)[-1]: target for target in other_targets}
    for path in component_modules(plugin):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_") and node.name in other_classes:
                findings.append(Finding("warning", "name-duplicate", path, f"{node.name} shares its name with {other_classes[node.name]}"))

    other_titles = {}
    for target in other_targets:
        origin = module_origin(target.rpartition(".")[0])
        if origin is not None:
            other_titles.update({(component_kind(target), title): target for title in source_titles(origin)})
    for path in component_modules(plugin):
        kind = component_kind(".".join(path.relative_to(plugin.root / "src").with_suffix("").parts))
        for title in source_titles(path):
            if (kind, title) in other_titles:
                title_key = (kind, title)
                findings.append(Finding("warning", "source-duplicate", path, f'cites "{title}" like {other_titles[title_key]}; reuse it if it is the same method'))
    return findings


def check_task_names(plugin: Plugin) -> list[Finding]:
    """A task preset's file is named after the task it selects, which must have a contract."""
    known = contract_tasks() | {p.stem.replace("_", "-") for p in (plugin.src / "task").glob("*.py") if not p.name.startswith("_")}
    return [
        Finding("error", "task-name", path, f"no task named {path.stem}; name the preset after the task it selects ({', '.join(sorted(known))})")
        for path in conf_files(plugin.conf / "task")
        if path.stem not in known
    ]


def check_provider(plugin: Plugin, composed: list[tuple[Path, str, Any]], *, publish: bool) -> list[Finding]:
    """Every ``teia.*`` module the plugin imports, targets, or mounts through a composed preset or module."""
    teia_root = _teia_root()
    level = "error" if publish else "warning"
    findings = []
    modules = [(path, name) for path in python_files(plugin) for name in imported_teia_modules(path.read_text(encoding="utf-8"))]
    modules += [(path, target.rpartition(".")[0]) for path, target in conf_targets(plugin.conf) if target.startswith("teia.")]
    for path, _, config in composed:
        if not isinstance(config, Exception):
            modules += [(path, target.rpartition(".")[0]) for target in _targets(config) if target.startswith("teia.")]
    for path, module in dict.fromkeys(modules):
        origin = module_origin(module)
        if origin is not None and not origin.is_relative_to(teia_root):
            findings.append(Finding(level, "provider", path, f"{module} comes from {provider_of(origin) or origin}, which is not the {DISTRIBUTION} distribution"))
    return findings


def compositions(plugin: Plugin) -> list[tuple[Path, str, Any]]:
    """Compose every task preset and module the plugin ships: ``(file, override, config or the exception raised)``."""
    from teia.core.runtime.composer import compose_project_config

    selections = [(path, f"task={path.relative_to(plugin.conf / 'task').with_suffix('').as_posix()}") for path in conf_files(plugin.conf / "task")]
    for group in MODULE_GROUPS:
        selections += [(path, f"{group}={path.relative_to(plugin.conf / group).with_suffix('').as_posix()}") for path in conf_files(plugin.conf / group)]
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for path, override in selections:
            try:
                results.append((path, override, compose_project_config(project_dir=Path(tmp), overrides=[override, "data_root=data"])))
            except Exception as exc:  # noqa: BLE001 - every failure is a finding
                results.append((path, override, exc))
    return results


def check_compose(composed: list[tuple[Path, str, Any]]) -> list[Finding]:
    return [Finding("error", "compose", path, f"{override} does not compose: {result}") for path, override, result in composed if isinstance(result, Exception)]


def check_lint(plugin: Plugin) -> list[Finding]:
    from teia.core.config.lint import run_lint

    with tempfile.TemporaryDirectory() as tmp:
        lint = run_lint(Path(tmp))
    own = plugin.conf.resolve()
    return [Finding("error", "lint", f.path, f"[{f.rule}] {f.message}") for f in lint if Path(f.path).resolve().is_relative_to(own)]


# -- orchestration -----------------------------------------------------------------------------------------------------


def check(root: Path, *, publish: bool = False, namespace: str | None = None) -> list[Finding]:
    plugin = load_plugin(root, namespace)
    shared = namespace == "teia"
    findings = check_pyproject(plugin) + check_license_file(plugin, publish=publish) + check_searchpath(plugin)
    findings += check_targets(plugin, shared=shared) + check_docstrings(plugin)
    if not shared:
        findings += check_package(plugin) + check_conf_owner(plugin)
    installed = check_installed(plugin)
    if installed:
        return findings + installed
    layers = conf_layers()
    findings += check_registered(plugin, layers) + check_imports(plugin) + check_target_imports(plugin) + check_base_classes(plugin)
    composed = compositions(plugin)
    findings += check_duplicates(plugin, layers) + check_task_names(plugin) + check_compose(composed) + check_lint(plugin)
    if not shared:
        findings += check_provider(plugin, composed, publish=publish)
    return findings


# -- helpers -----------------------------------------------------------------------------------------------------------


def _teia_root() -> Path:
    return Path(importlib.import_module("teia.core").__file__).resolve().parents[1]


def conf_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.yaml")) if root.is_dir() else []


def conf_targets(root: Path) -> Iterator[tuple[Path, str]]:
    for path in conf_files(root):
        yield from ((path, target) for target in _targets(yaml.safe_load(path.read_text(encoding="utf-8"))))


def _targets(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str) and "${" not in target:
            yield target
        for child in value.values():
            yield from _targets(child)
    elif isinstance(value, list):
        for child in value:
            yield from _targets(child)


def python_files(plugin: Plugin) -> list[Path]:
    files = sorted(plugin.src.rglob("*.py")) if plugin.src.is_dir() else []
    return files + sorted((plugin.root / "hydra_plugins").glob("*.py"))


def component_modules(plugin: Plugin) -> list[Path]:
    return sorted(p for sub in ("node", "task") for p in (plugin.src / sub).rglob("*.py") if p.name != "protocol.py")


def module_source(plugin: Plugin, module: str) -> Path | None:
    relative = Path(*module.split("."))
    for candidate in (plugin.root / "src" / relative.with_suffix(".py"), plugin.root / "src" / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def defined_names(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


def imported_tops(source: str) -> set[str]:
    tops = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            tops.add(node.module.split(".")[0])
    return tops


def imported_teia_modules(source: str) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("teia."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.startswith("teia."):
            modules.add(node.module)
    return modules


def component_kind(dotted: str) -> tuple[str, ...]:
    """``(graph, kind)`` of a component path: ``pkg.node.net.loss.x.Y`` -> ``("net", "loss")``."""
    parts = dotted.split(".")
    return tuple(parts[parts.index("node") + 1 : parts.index("node") + 3]) if "node" in parts else tuple(parts[1:2])


def source_titles(path: Path) -> set[str]:
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    return {title for title in _TITLE.findall(doc) if title.lower() != "common knowledge"}


def resolve_target(target: str) -> Any:
    module, _, symbol = target.rpartition(".")
    return getattr(importlib.import_module(module), symbol)


def module_origin(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return Path(spec.origin).resolve() if spec and spec.has_location else None


def contract_tasks(exclude: Path | None = None) -> set[str]:
    import teia.task

    names = set()
    for info in pkgutil.iter_modules(teia.task.__path__):
        if not info.name.startswith("_") and not (exclude and Path(info.module_finder.path).resolve().is_relative_to(exclude.resolve())):
            names.add(info.name.replace("_", "-"))
    return names


def conf_layers() -> list[tuple[str, Path]]:
    from teia.core.config.layers import core_conf_layer, plugin_conf_layers

    return [(layer.label, layer.path.resolve()) for layer in [core_conf_layer(), *plugin_conf_layers()]]


def declared_distributions(plugin: Plugin) -> list[str]:
    project = plugin.pyproject.get("project", {})
    extras = [req for reqs in project.get("optional-dependencies", {}).values() for req in reqs]
    return _requirement_names([*project.get("dependencies", []), *extras])


def _requirement_names(requirements: list[str]) -> list[str]:
    return [match.group(1) for req in requirements if (match := _NAME.match(req))]


@lru_cache(maxsize=1)
def teia_requirements() -> tuple[str, ...]:
    return tuple(_requirement_names([req for req in metadata.requires(DISTRIBUTION) or [] if "extra ==" not in req]))


def editable_root(dist: metadata.Distribution) -> Path | None:
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    url = json.loads(text).get("url", "")
    return Path(url.removeprefix("file://")).resolve() if url.startswith("file://") else None


@lru_cache(maxsize=1)
def _provider_roots() -> tuple[tuple[str, Path], ...]:
    roots = []
    for dist in metadata.distributions():
        root = editable_root(dist)
        if root is not None:
            roots.append((dist.metadata["Name"], root))
    return tuple(sorted(roots, key=lambda item: len(item[1].parts), reverse=True))


def provider_of(path: Path) -> str | None:
    """Name of the installed distribution a file belongs to, or ``None``."""
    for name, root in _provider_roots():
        if path.is_relative_to(root):
            return name
    for dist in metadata.distributions():
        if any(Path(dist.locate_file(f)).resolve() == path for f in dist.files or []):
            return dist.metadata["Name"]
    return None


# -- scaffold ----------------------------------------------------------------------------------------------------------

_PYPROJECT = """\
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.10"
license = "{license}"
dependencies = ["{distribution}>={teia_version}"]

[tool.setuptools.packages.find]
where = ["src", "."]
include = ["{package}*", "hydra_plugins*"]

[tool.setuptools.package-data]
"*" = ["**/*.yaml"]
"{package}" = ["skills/**/*"]
"""

_SEARCHPATH = '''"""Registers {package}'s conf/ on the Hydra search path, so `teia` sees its options after install."""

from importlib.resources import files

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class {cls}SearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="{package}", path=f"file://{{files('{package}') / 'conf'}}")
'''


def scaffold(root: Path, name: str, license: str = "Apache-2.0") -> list[tuple[Path, bool]]:
    """Create the files a plugin needs (specs/plugin.md). Existing files are kept: ``(path, created)`` per file."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise SystemExit(f"{name!r} must be a lowercase distribution name ([a-z][a-z0-9_-]*).")
    package = name.replace("-", "_")
    files = {
        root / "pyproject.toml": _PYPROJECT.format(name=name, package=package, license=license, distribution=DISTRIBUTION, teia_version=metadata.version(DISTRIBUTION)),
        root / "hydra_plugins" / f"{package}_searchpath.py": _SEARCHPATH.format(package=package, cls="".join(p.title() for p in package.split("_"))),
        root / "src" / package / "__init__.py": "",
        root / "src" / package / "conf" / ".gitkeep": "",
        root / "src" / package / "node" / ".gitkeep": "",
    }
    created = []
    for path, text in files.items():
        if path.exists():
            created.append((path, False))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        created.append((path, True))
    return created
