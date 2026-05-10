from __future__ import annotations

"""
Async MCP client — stdio JSON-RPC 2.0 transport.

Spawns an MCP server as a subprocess and communicates via stdin/stdout.
Each JSON-RPC request gets a unique ID; responses are matched by ID via
asyncio.Future so concurrent calls are safe.

Usage:
    client = MCPClient("github", "npx", ["-y", "@modelcontextprotocol/server-github"],
                       env={"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."})
    await client.start()
    tools = await client.list_tools()
    result = await client.call_tool("create_issue", {"owner": "...", "repo": "...", ...})
    await client.close()
"""

import asyncio
import json
import logging
import os
from typing import Any

log = logging.getLogger("axor.mcp.client")

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    def __init__(self, error: dict) -> None:
        super().__init__(error.get("message", str(error)))
        self.code = error.get("code")
        self.data = error.get("data")


class MCPClient:
    """
    Manages a single MCP server subprocess over stdio transport.

    Thread-safe: all operations are coroutines running in the same event loop.
    The internal reader task dispatches responses to waiting futures by ID.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        self.name    = name
        self._command = command
        self._args    = args
        self._env     = env or {}
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._started = False

    # ── Public API ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._started:
            return
        merged_env = {**os.environ, **self._env}
        self._proc = await asyncio.create_subprocess_exec(
            self._command, *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=merged_env,
        )
        self._reader = asyncio.create_task(self._read_loop(), name=f"mcp-{self.name}")
        await self._request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities":    {},
            "clientInfo":      {"name": "axor", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized")
        self._started = True
        log.debug("MCP server '%s' started", self.name)

    async def list_tools(self) -> list[dict]:
        result = await self._request("tools/list", {})
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        result = await self._request("tools/call", {
            "name":      tool_name,
            "arguments": arguments,
        })
        if result.get("isError"):
            return f"MCP error: {_extract_text(result)}"
        return _extract_text(result)

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
        if self._proc is not None and self._proc.stdin:
            try:
                self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except Exception:
                self._proc.kill()

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                self._dispatch(line)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning("MCP reader error for '%s': %s", self.name, exc)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError(f"MCP server '{self.name}' disconnected"))
            self._pending.clear()

    def _dispatch(self, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return
        msg_id = msg.get("id")
        if msg_id is None:
            return  # notification — ignore
        fut = self._pending.pop(msg_id, None)
        if fut is None or fut.done():
            return
        if "error" in msg:
            fut.set_exception(MCPError(msg["error"]))
        else:
            fut.set_result(msg.get("result") or {})

    async def _request(self, method: str, params: dict) -> dict:
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        await self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        return await asyncio.wait_for(fut, timeout=30.0)

    async def _notify(self, method: str, params: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params:
            msg["params"] = params
        await self._send(msg)

    async def _send(self, msg: dict) -> None:
        assert self._proc and self._proc.stdin
        data = json.dumps(msg, ensure_ascii=False) + "\n"
        self._proc.stdin.write(data.encode())
        await self._proc.stdin.drain()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_text(result: dict) -> str:
    parts: list[str] = []
    for block in result.get("content", []):
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "image":
            parts.append("[image]")
        elif btype == "resource":
            parts.append(block.get("text") or str(block.get("resource", "")))
    return "\n".join(parts) if parts else ""
