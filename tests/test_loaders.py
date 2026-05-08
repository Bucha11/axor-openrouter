"""Tests for AxorSkillLoader."""
from __future__ import annotations

import pathlib
import pytest

from axor_openrouter.loaders import AxorSkillLoader


def test_no_files_returns_empty(tmp_path):
    loader = AxorSkillLoader(root=tmp_path)
    assert loader.load() == []


def test_claude_md_returned(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# Project guide")
    loader = AxorSkillLoader(root=tmp_path)
    docs = loader.load()
    assert len(docs) == 1
    assert "Project guide" in docs[0]


def test_skills_dir_files_returned(tmp_path):
    skills = tmp_path / ".axor" / "skills"
    skills.mkdir(parents=True)
    (skills / "a.md").write_text("skill A")
    (skills / "b.md").write_text("skill B")
    loader = AxorSkillLoader(root=tmp_path)
    docs = loader.load()
    assert len(docs) == 2
    assert any("skill A" in d for d in docs)
    assert any("skill B" in d for d in docs)


def test_both_claude_md_and_skills(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# Guide")
    skills = tmp_path / ".axor" / "skills"
    skills.mkdir(parents=True)
    (skills / "x.md").write_text("extra skill")
    loader = AxorSkillLoader(root=tmp_path)
    docs = loader.load()
    assert len(docs) == 2


def test_docs_sorted_by_path(tmp_path):
    skills = tmp_path / ".axor" / "skills"
    skills.mkdir(parents=True)
    (skills / "z.md").write_text("last")
    (skills / "a.md").write_text("first")
    loader = AxorSkillLoader(root=tmp_path)
    docs = loader.load()
    assert docs[0] == "first"
    assert docs[1] == "last"


def test_unreadable_file_skipped(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("good")
    # skills dir but no files
    loader = AxorSkillLoader(root=tmp_path)
    docs = loader.load()
    assert len(docs) == 1
