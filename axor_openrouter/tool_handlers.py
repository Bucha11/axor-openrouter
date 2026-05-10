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
    @property
    def name(self) -> str:
        return "search"

    async def execute(self, args: dict[str, Any]) -> Any:
        pattern = args["pattern"]
        path = args.get("path", ".")
        search_type = args.get("type", "both")
        results = []

        if search_type in ("file", "both"):
            proc = await asyncio.create_subprocess_shell(
                f"find {path} -name '{pattern}' 2>/dev/null | head -50",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if stdout.strip():
                results.append("Files:\n" + stdout.decode(errors="replace"))

        if search_type in ("content", "both"):
            proc = await asyncio.create_subprocess_shell(
                f"grep -r --include='*' '{pattern}' {path} 2>/dev/null | head -50",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            if stdout.strip():
                results.append("Content matches:\n" + stdout.decode(errors="replace"))

        return "\n".join(results) if results else "No results found."


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


_HANDLER_MAP: dict[str, type[ToolHandler]] = {
    "read":   ReadHandler,
    "write":  WriteHandler,
    "edit":   EditHandler,
    "bash":   BashHandler,
    "search": SearchHandler,
    "glob":   GlobHandler,
}


def make_capability_executor(tools: tuple[str, ...]) -> CapabilityExecutor:
    cap = CapabilityExecutor()
    for name in tools:
        cls = _HANDLER_MAP.get(name)
        if cls is not None:
            cap.register(cls())
    return cap
