from __future__ import annotations

import asyncio
import glob as glob_module
import os
from typing import Any

from axor_core.capability.executor import CapabilityExecutor, ToolHandler


def _get_path(args: dict[str, Any]) -> str:
    """Accept path under 'path', 'file_path', or 'filename' keys."""
    return args.get("path") or args.get("file_path") or args.get("filename") or args.get("filepath") or ""


class ReadHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args: dict[str, Any]) -> Any:
        path = _get_path(args)
        if not path:
            return "Error: no path argument provided"
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"


class WriteHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "write"

    async def execute(self, args: dict[str, Any]) -> Any:
        path = _get_path(args)
        content = args.get("content") or args.get("text") or args.get("data") or ""
        if not path:
            return "Error: no path argument provided"
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"


class BashHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "bash"

    async def execute(self, args: dict[str, Any]) -> Any:
        command = args["command"]
        timeout = int(args.get("timeout", 30))
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            return output or f"(exit code {proc.returncode})"
        except asyncio.TimeoutError:
            return f"Command timed out after {timeout}s"


class SearchHandler(ToolHandler):
    """
    Content and file search.

    Content search backend priority:
      1. ripgrep (rg) — fast, follows symlinks, respects .gitignore
      2. grep -R      — follows symlinks, universal fallback

    File search uses find -L (follows symlinks).
    """

    @property
    def name(self) -> str:
        return "search"

    async def execute(self, args: dict[str, Any]) -> Any:
        pattern     = args["pattern"]
        path        = args.get("path", ".")
        search_type = args.get("type", "both")
        include     = args.get("include", "")        # e.g. "*.py"
        context     = min(int(args.get("context", 0)), 10)
        ignore_case = bool(args.get("ignore_case", False))

        results: list[str] = []

        if search_type in ("file", "both"):
            out = await self._find_files(pattern, path)
            if out:
                results.append("Files:\n" + out)

        if search_type in ("content", "both"):
            out = await self._search_content(pattern, path, include, context, ignore_case)
            if out:
                results.append("Content matches:\n" + out)

        return "\n".join(results) if results else "No results found."

    async def _find_files(self, pattern: str, path: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "find", "-L", path, "-name", pattern,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        lines = stdout.decode(errors="replace").splitlines()
        return "\n".join(lines[:100])

    async def _search_content(
        self, pattern: str, path: str, include: str, context: int, ignore_case: bool
    ) -> str:
        try:
            return await self._rg(pattern, path, include, context, ignore_case)
        except FileNotFoundError:
            # rg not installed — fall back to grep
            return await self._grep(pattern, path, include, context, ignore_case)

    async def _rg(
        self, pattern: str, path: str, include: str, context: int, ignore_case: bool
    ) -> str:
        cmd = ["rg", "--follow", "--no-ignore", "--line-number", "--no-heading", "--color=never"]
        if ignore_case:
            cmd.append("--ignore-case")
        if context > 0:
            cmd += ["-C", str(context)]
        if include:
            cmd += ["--glob", include]
        cmd += ["--", pattern, path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        # rg exits 1 on no matches — that's not an error
        lines = stdout.decode(errors="replace").splitlines()
        return "\n".join(lines[:200])

    async def _grep(
        self, pattern: str, path: str, include: str, context: int, ignore_case: bool
    ) -> str:
        # -R follows symlinks (unlike -r)
        cmd = ["grep", "-R", "--line-number", "--color=never"]
        if ignore_case:
            cmd.append("--ignore-case")
        if context > 0:
            cmd += ["-C", str(context)]
        if include:
            cmd += [f"--include={include}"]
        cmd += ["--", pattern, path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        lines = stdout.decode(errors="replace").splitlines()
        return "\n".join(lines[:200])


class EditHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "edit"

    async def execute(self, args: dict[str, Any]) -> Any:
        path = _get_path(args)
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        replace_all = bool(args.get("replace_all", False))

        if not path:
            return "Error: no path argument provided"
        if not old_string:
            return "Error: old_string is required"

        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            return f"Error: file not found: {path}"
        except PermissionError:
            return f"Error: permission denied: {path}"

        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string found {count} times — use replace_all=true "
                "or provide more surrounding context to make it unique"
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {path}"


class GlobHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "glob"

    async def execute(self, args: dict[str, Any]) -> Any:
        pattern = args["pattern"]
        cwd = args.get("cwd", ".")
        full_pattern = os.path.join(cwd, pattern)
        matches = glob_module.glob(full_pattern, recursive=True)
        return "\n".join(matches) if matches else "No matches found."


class WebFetchHandler(ToolHandler):
    """
    Fetch the content of a URL. Returns up to max_bytes of the response body.

    Args:
        url          URL to fetch (http/https)
        max_bytes    Max bytes to return (default 65536 = 64 KB)
        timeout      Request timeout in seconds (default 15)
    """

    _MAX_BYTES_DEFAULT = 65_536  # 64 KB

    @property
    def name(self) -> str:
        return "fetch"

    async def execute(self, args: dict[str, Any]) -> Any:
        import urllib.request
        import urllib.error

        url: str = args.get("url") or args.get("uri") or ""
        if not url:
            return "Error: 'url' argument is required."
        if not url.startswith(("http://", "https://")):
            return f"Error: only http/https URLs are supported (got {url!r})."

        max_bytes: int = int(args.get("max_bytes", self._MAX_BYTES_DEFAULT))
        timeout: int = int(args.get("timeout", 15))

        def _get() -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "axor/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read(max_bytes)
                    charset = resp.headers.get_content_charset() or "utf-8"
                    content = raw.decode(charset, errors="replace")
                    truncated = len(raw) >= max_bytes
                    ct = resp.headers.get_content_type() or ""
                    header = f"[{resp.status} {url}  content-type: {ct}]\n"
                    suffix = f"\n[truncated at {max_bytes} bytes]" if truncated else ""
                    return header + content + suffix
            except urllib.error.HTTPError as e:
                return f"HTTP {e.code}: {e.reason}  ({url})"
            except urllib.error.URLError as e:
                return f"URL error: {e.reason}  ({url})"

        try:
            return await asyncio.wait_for(asyncio.to_thread(_get), timeout=timeout + 2)
        except asyncio.TimeoutError:
            return f"Timeout fetching {url} after {timeout}s"


class TodoStore:
    """Session-scoped todo list shared between TodoWriteHandler and TodoReadHandler."""

    _STATUS_ICON = {"pending": "○", "in_progress": "◉", "completed": "✓"}
    _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

    def __init__(self) -> None:
        self._todos: list[dict] = []

    def write(self, todos: list[dict]) -> None:
        self._todos = [
            {
                "id":       str(t.get("id", i + 1)),
                "content":  str(t.get("content", "")),
                "status":   t.get("status", "pending"),
                "priority": t.get("priority", "medium"),
            }
            for i, t in enumerate(todos)
        ]

    def read(self) -> list[dict]:
        return list(self._todos)

    def format(self) -> str:
        if not self._todos:
            return "No todos."
        order = self._PRIORITY_ORDER
        sorted_todos = sorted(
            self._todos,
            key=lambda t: (order.get(t.get("priority", "medium"), 1), t["id"]),
        )
        lines = []
        for t in sorted_todos:
            icon = self._STATUS_ICON.get(t["status"], "○")
            pri = t["priority"]
            tag = f"[{pri}] " if pri != "medium" else ""
            lines.append(f"  {icon} {tag}{t['content']}")
        return "\n".join(lines)


class TodoWriteHandler(ToolHandler):
    """
    Replace the session's todo list.

    Pass the complete updated list each time — this is a full replace, not a patch.
    Status values: pending | in_progress | completed
    Priority values: high | medium | low
    """

    def __init__(self, store: TodoStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "todo_write"

    async def execute(self, args: dict[str, Any]) -> Any:
        todos = args.get("todos", [])
        if not isinstance(todos, list):
            return "Error: 'todos' must be a list of objects."
        self._store.write(todos)
        return self._store.format()


class TodoReadHandler(ToolHandler):
    """Return the current session todo list."""

    def __init__(self, store: TodoStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "todo_read"

    async def execute(self, args: dict[str, Any]) -> Any:
        return self._store.format()


_HANDLER_MAP: dict[str, type[ToolHandler]] = {
    "read":   ReadHandler,
    "write":  WriteHandler,
    "edit":   EditHandler,
    "bash":   BashHandler,
    "search": SearchHandler,
    "glob":   GlobHandler,
    "fetch":  WebFetchHandler,
    # todo_write / todo_read are NOT in this map — they need a shared TodoStore
    # instance and are registered manually in make_session().
}


def make_capability_executor(tools: tuple[str, ...]) -> CapabilityExecutor:
    cap = CapabilityExecutor()
    for name in tools:
        cls = _HANDLER_MAP.get(name)
        if cls is not None:
            cap.register(cls())
    return cap
