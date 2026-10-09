from __future__ import annotations

import argparse
from pathlib import Path


COMMAND_NAME = "init"
HELP = "Initialize a teia project: create conf/ and link the skills of teia and every installed plugin for coding agents."

AGENT_DIRS = (".claude", ".agents")
GITIGNORE_HEADER = "# teia-managed skill links (absolute symlinks into the installed package; `teia init` recreates them)"


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(COMMAND_NAME, help=HELP)
    parser.add_argument("--project-root", default=".", help="defaults to the current directory")
    parser.set_defaults(_teia_handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    conf_dir = project_root / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)

    linked = _link_skills(project_root)
    _gitignore_links(project_root, linked)

    print(f"Initialized teia project at {project_root}")
    print(f"Skills linked: {', '.join(sorted({link.name for link in linked}))}")
    print("Next: `teia config list`, then `teia config init` to scaffold override stubs in conf/.")
    return 0


def _skills_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "skills"
    if not root.is_dir():
        raise FileNotFoundError(f"teia skills directory is missing from the installation: {root}")
    return root


def _skills_roots() -> list[Path]:
    """teia's own skills, then those of every installed plugin: a ``skills/`` folder beside its registered ``conf/``."""
    from teia.core.config.layers import plugin_conf_layers

    roots = [_skills_root(), *(layer.path.parent / "skills" for layer in plugin_conf_layers())]
    return list(dict.fromkeys(root.resolve() for root in roots if root.is_dir()))


def _link_skills(project_root: Path) -> list[Path]:
    """Symlink each ``skills/<name>/`` holding a ``SKILL.md`` into every agent skills dir.

    Links are absolute so a pip-installed package stays the single source of truth. Dangling
    symlinks left by removed skills are dropped, and a real file or directory is never replaced.
    """
    skill_dirs = sorted((p for root in _skills_roots() for p in root.iterdir() if (p / "SKILL.md").is_file()), key=lambda p: p.name)
    names = [p.name for p in skill_dirs]
    clashes = sorted({name for name in names if names.count(name) > 1})
    if clashes:
        raise ValueError(f"two installed packages ship a skill with the same name: {clashes}")
    linked: list[Path] = []
    for agent_dir in AGENT_DIRS:
        dest = project_root / agent_dir / "skills"
        dest.mkdir(parents=True, exist_ok=True)

        for stale in (p for p in dest.iterdir() if p.is_symlink() and not p.exists()):
            stale.unlink()

        for skill_dir in skill_dirs:
            link = dest / skill_dir.name
            if link.is_symlink():
                link.unlink()
            elif link.exists():
                print(f"Warning: {link} already exists and is not a symlink managed by teia.")
                continue
            link.symlink_to(skill_dir)
            linked.append(link)
    return linked


def _gitignore_links(project_root: Path, links: list[Path]) -> None:
    gitignore = project_root / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
    entries = [link.relative_to(project_root).as_posix() for link in links]
    missing = [entry for entry in entries if entry not in existing]
    if not missing:
        return
    header = [] if GITIGNORE_HEADER in existing else [GITIGNORE_HEADER]
    text = "\n".join([*existing, *header, *missing]) + "\n"
    gitignore.write_text(text, encoding="utf-8")
