from __future__ import annotations

import asyncio
from typing import Any


class ToolResultBus:
    """
    Async channel for passing tool results from IntentLoop back to the executor.

    Protocol (mirrors ClaudeCodeExecutor pattern expected by wrapper.py):

        wrapper.py detects get_bus() on the executor and registers a callback:

            async def _push_to_bus(tool_use_id, tool_name, result, approved):
                bus.push(tool_use_id, result)   # synchronous put_nowait

        The executor yields a TOOL_USE event, which pauses the generator.
        The intent_loop catches the event, executes the tool, calls the callback
        (which puts the result into the queue), then resumes the generator.
        The generator does `await bus.wait()` — returns immediately since the
        result is already in the queue.

    This is correct even with multiple sequential tool calls because each
    yield/wait pair is sequential and the queue is FIFO.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def push(self, tool_use_id: str, result: Any) -> None:
        """Called synchronously by intent_loop callback (put_nowait is safe)."""
        self._queue.put_nowait((tool_use_id, result))

    async def wait(self) -> tuple[str, Any]:
        """Called by executor to receive the next tool result."""
        return await self._queue.get()

    def clear(self) -> None:
        """Drain queue — call between turns if needed."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
