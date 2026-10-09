from __future__ import annotations

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from teia.core.commands import init


def _init(project: Path, skills: Path | list[Path]) -> str:
    out = io.StringIO()
    with mock.patch.object(init, "_skills_roots", return_value=list(skills) if isinstance(skills, list) else [skills]), contextlib.redirect_stdout(out):
        init.run(argparse.Namespace(project_root=str(project)))
    return out.getvalue()


class TeiaInitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.skills = tmp / "skills"
        (self.skills / "teia").mkdir(parents=True)
        (self.skills / "teia" / "SKILL.md").write_text("---\nname: teia\n---\n")
        (self.skills / "not-a-skill").mkdir()
        self.project = tmp / "project"
        self.project.mkdir()

    def test_links_only_skill_dirs_into_both_agent_dirs_and_creates_conf(self) -> None:
        _init(self.project, self.skills)
        for agent in (".claude", ".agents"):
            link = self.project / agent / "skills" / "teia"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (self.skills / "teia").resolve())
            self.assertFalse((self.project / agent / "skills" / "not-a-skill").exists())
        self.assertTrue((self.project / "conf").is_dir())

    def test_rerun_is_idempotent_and_gitignore_is_not_duplicated(self) -> None:
        _init(self.project, self.skills)
        first = (self.project / ".gitignore").read_text()
        _init(self.project, self.skills)
        self.assertEqual((self.project / ".gitignore").read_text(), first)
        self.assertIn(".claude/skills/teia", first.splitlines())
        self.assertIn(".agents/skills/teia", first.splitlines())

    def test_gitignore_keeps_existing_lines(self) -> None:
        (self.project / ".gitignore").write_text("venv/\n")
        _init(self.project, self.skills)
        self.assertEqual((self.project / ".gitignore").read_text().splitlines()[0], "venv/")

    def test_dangling_symlinks_are_removed(self) -> None:
        dest = self.project / ".claude" / "skills"
        dest.mkdir(parents=True)
        (dest / "auto-research").symlink_to(self.project / "gone")
        _init(self.project, self.skills)
        self.assertFalse((dest / "auto-research").is_symlink())

    def test_real_directory_is_kept_and_not_ignored(self) -> None:
        mine = self.project / ".claude" / "skills" / "teia"
        mine.mkdir(parents=True)
        out = _init(self.project, self.skills)
        self.assertIn("not a symlink managed by teia", out)
        self.assertFalse(mine.is_symlink())
        self.assertNotIn(".claude/skills/teia", (self.project / ".gitignore").read_text().splitlines())

    def test_links_the_skills_of_every_root_and_rejects_name_clashes(self) -> None:
        plugin = self.project.parent / "plugin_skills"
        (plugin / "extra").mkdir(parents=True)
        (plugin / "extra" / "SKILL.md").write_text("---\nname: extra\n---\n")
        _init(self.project, [self.skills, plugin])
        self.assertTrue((self.project / ".claude" / "skills" / "extra").is_symlink())
        (plugin / "teia").mkdir()
        (plugin / "teia" / "SKILL.md").write_text("---\nname: teia\n---\n")
        with self.assertRaises(ValueError):
            _init(self.project, [self.skills, plugin])

    def test_missing_skills_dir_fails_fast(self) -> None:
        with mock.patch.object(init, "__file__", str(self.project / "a" / "b" / "c.py")):
            with self.assertRaises(FileNotFoundError):
                init._skills_root()


if __name__ == "__main__":
    unittest.main()
