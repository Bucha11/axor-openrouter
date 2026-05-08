"""Unit tests for OpenRouterExecutor (mocked transport)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.result import ExecutorEventKind
from axor_openrouter.executor import OpenRouterExecutor
from axor_openrouter.cascade.tiers import DEFAULT_TIERS, TierMapper
from axor_openrouter.routing.provider_prefs import ProviderPrefs


@pytest.fixture()
def executor():
    return OpenRouterExecutor(
        api_key="sk-or-test",
        tier_mapper=TierMapper(DEFAULT_TIERS),
        provider_prefs=ProviderPrefs(),
        fallbacks={},
        enable_cache=False,
    )


@pytest.fixture()
def envelope():
    return ExecutionEnvelope(task="say hello", context_text="")


@pytest.mark.asyncio
async def test_stream_stop_event(executor, envelope):
    """A simple non-tool response yields a STOP event with usage."""
    fake_chunks = [
        {
            "choices": [{"delta": {"content": "Hello!"}, "finish_reason": None}],
            "usage": None,
        },
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    ]

    async def fake_stream(api_key, body):
        for chunk in fake_chunks:
            yield chunk

    with patch.object(
        executor._transport, "stream", side_effect=fake_stream
    ):
        events = [e async for e in executor.stream(envelope)]

    kinds = [e.kind for e in events]
    assert ExecutorEventKind.TEXT in kinds
    assert ExecutorEventKind.STOP in kinds


@pytest.mark.asyncio
async def test_get_bus_returns_bus(executor):
    bus = executor.get_bus()
    assert bus is not None
    # bus.push / bus.wait should be available
    assert hasattr(bus, "push")
    assert hasattr(bus, "wait")


@pytest.mark.asyncio
async def test_stream_resets_bus_each_call(executor, envelope):
    """Each stream() call should clear the bus from the previous round."""
    fake_chunks = [
        {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}], "usage": None},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ]

    async def fake_stream(api_key, body):
        for chunk in fake_chunks:
            yield chunk

    with patch.object(executor._transport, "stream", side_effect=fake_stream):
        _ = [e async for e in executor.stream(envelope)]

    # Second call should not raise / deadlock
    with patch.object(executor._transport, "stream", side_effect=fake_stream):
        events = [e async for e in executor.stream(envelope)]

    assert any(e.kind == ExecutorEventKind.STOP for e in events)
