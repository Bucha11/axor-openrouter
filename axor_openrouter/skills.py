from __future__ import annotations

"""
Generic skill and plugin loader for axor-openrouter.

Reads project-level instructions and skills from the filesystem,
packages them as ExtensionBundle for GovernedSession.

Discovery order (all optional, missing paths silently skipped):
  1. ~/.claude/CLAUDE.md          — user-level global instructions
  2. <cwd>/CLAUDE.md              — project-level instructions
  3. <cwd>/.claude/skills/*.md    — individual skills
"""

import logging
import os
from pathlib import Path

from axor_core.contracts.extension import (
    ExtensionBundle,
    ExtensionFragment,
    ExtensionLoader,
)

log = logging.getLogger("axor.openrouter.skills")

# Chars per fragment — sanitizer enforces 2000 token ≈ 8000 char ceiling,
# but we soft-cap here to keep context concise.
_MAX_FRAGMENT_CHARS = 6000


class GenericSkillLoader(ExtensionLoader):
    """
    Loads CLAUDE.md + .claude/skills/*.md into the session context.

    Model-agnostic — works with DeepSeek, GPT, llama, or any OpenRouter model.
    The fragments are injected as plain text into ContextView before each task.
    """

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._cwd = Path(cwd) if cwd else Path.cwd()

    async def load(self) -> ExtensionBundle:
        fragments: list[ExtensionFragment] = []

        for path, name in self._candidate_files():
            text = _read_file(path)
            if text:
                fragments.append(ExtensionFragment(
                    name=name,
                    context_fragment=text,
                    required_tools=(),
                    policy_overrides={},
                    source=str(path),
                ))
                log.debug("skill loaded: %s (%d chars)", name, len(text))

        return ExtensionBundle(fragments=tuple(fragments))

    def _candidate_files(self) -> list[tuple[Path, str]]:
        candidates: list[tuple[Path, str]] = []

        # 1. User-level global instructions
        user_claude_md = Path.home() / ".claude" / "CLAUDE.md"
        candidates.append((user_claude_md, "user/CLAUDE.md"))

        # 2. Project-level instructions
        project_claude_md = self._cwd / "CLAUDE.md"
        candidates.append((project_claude_md, "CLAUDE.md"))

        # 3. Individual skills under .claude/skills/
        skills_dir = self._cwd / ".claude" / "skills"
        if skills_dir.is_dir():
            for skill_file in sorted(skills_dir.glob("*.md")):
                candidates.append((skill_file, f"skill/{skill_file.stem}"))

        return candidates


def _read_file(path: Path) -> str:
    """Read a file, returning empty string on any error or if it's empty."""
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > _MAX_FRAGMENT_CHARS:
            text = text[:_MAX_FRAGMENT_CHARS] + "\n[... truncated]"
        return text
    except OSError as e:
        log.warning("could not read skill file %s: %s", path, e)
        return ""
