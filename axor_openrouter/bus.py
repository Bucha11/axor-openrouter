from __future__ import annotations

import asyncio
from typing import Any


class ToolResultBus:
    """
    Async channel for passing tool results from IntentLoop back to the executor.

    wrapper.py detects get_bus() on the executor and registers a callback:

        async def _push_to_bus(tool_use_id, tool_name, result, approved):
            bus.push(tool_use_id, result)

    The executor yields a TOOL_USE event, awaits bus.wait(), and gets the result.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def push(self, tool_use_id: str, result: Any) -> None:
        self._queue.put_nowait((tool_use_id, result))

    async def wait(self) -> tuple[str, Any]:
        return await self._queue.get()

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
