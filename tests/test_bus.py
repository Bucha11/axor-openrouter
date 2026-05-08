"""Tests for ToolResultBus."""
from __future__ import annotations

import asyncio
import pytest

from axor_openrouter.bus import ToolResultBus


@pytest.mark.asyncio
async def test_push_wait_single():
    bus = ToolResultBus()
    bus.push("id1", "result_a")
    tid, result = await bus.wait()
    assert tid == "id1"
    assert result == "result_a"


@pytest.mark.asyncio
async def test_push_wait_fifo_order():
    bus = ToolResultBus()
    bus.push("id1", "r1")
    bus.push("id2", "r2")
    assert await bus.wait() == ("id1", "r1")
    assert await bus.wait() == ("id2", "r2")


@pytest.mark.asyncio
async def test_push_dict_result():
    bus = ToolResultBus()
    bus.push("id1", {"content": "file", "lines": 42})
    _, result = await bus.wait()
    assert result["lines"] == 42


def test_clear_drains_queue():
    bus = ToolResultBus()
    bus.push("id1", "r1")
    bus.push("id2", "r2")
    bus.clear()
    assert bus._queue.qsize() == 0


@pytest.mark.asyncio
async def test_push_from_concurrent_coroutine():
    bus = ToolResultBus()
    results: list = []

    async def waiter():
        results.append(await bus.wait())

    async def pusher():
        await asyncio.sleep(0)
        bus.push("id_x", "payload")

    await asyncio.gather(waiter(), pusher())
    assert results == [("id_x", "payload")]
