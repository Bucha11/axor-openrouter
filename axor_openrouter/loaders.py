"""AxorSkillLoader: reads CLAUDE.md and .axor/skills/*.md as extra context."""
from __future__ import annotations

import pathlib
from typing import Iterator


class AxorSkillLoader:
    """Collects freeform skill/context documents from the project tree.

    Documents are injected as system-message fragments when building the
    request envelope so the model is always aware of project conventions.
    """

    def __init__(self, root: str | pathlib.Path = ".") -> None:
        self._root = pathlib.Path(root).resolve()

    def load(self) -> list[str]:
        """Return a list of document strings (deduplicated, sorted)."""
        docs: dict[pathlib.Path, str] = {}
        for path in self._candidate_paths():
            if path.is_file():
                try:
                    docs[path] = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
        return [docs[k] for k in sorted(docs)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _candidate_paths(self) -> Iterator[pathlib.Path]:
        # CLAUDE.md at the project root
        yield self._root / "CLAUDE.md"
        # .axor/skills/*.md
        skills_dir = self._root / ".axor" / "skills"
        if skills_dir.is_dir():
            yield from sorted(skills_dir.glob("*.md"))
