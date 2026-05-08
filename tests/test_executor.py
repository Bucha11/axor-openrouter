"""Tests for OpenRouterExecutor."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from axor_core.contracts.result import ExecutorEventKind


# ── Helpers ───────────────────────────────────────────────────────────────────

_STOP_CHUNKS = [
    {"choices": [{"delta": {"content": "Hello!"}, "finish_reason": None}], "usage": None},
    {"choices": [{"delta": {}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
]


async def _collect(gen):
    return [e async for e in gen]


# ── Basic streaming ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_event_is_routing_text(executor, envelope):
    async def fake_stream(api_key, body):
        for chunk in _STOP_CHUNKS:
            yield chunk

    with patch.object(executor._transport, "stream", side_effect=fake_stream):
        events = await _collect(executor.stream(envelope))

    first = events[0]
    assert first.kind == ExecutorEventKind.TEXT
    assert "_routing" in first.payload


@pytest.mark.asyncio
async def test_stream_yields_text_and_stop(executor, envelope):
    async def fake_stream(api_key, body):
        for chunk in _STOP_CHUNKS:
            yield chunk

    with patch.object(executor._transport, "stream", side_effect=fake_stream):
        events = await _collect(executor.stream(envelope))

    kinds = [e.kind for e in events]
    assert ExecutorEventKind.TEXT in kinds
    assert ExecutorEventKind.STOP in kinds


@pytest.mark.asyncio
async def test_stop_event_has_usage(executor, envelope):
    async def fake_stream(api_key, body):
        for chunk in _STOP_CHUNKS:
            yield chunk

    with patch.object(executor._transport, "stream", side_effect=fake_stream):
        events = await _collect(executor.stream(envelope))

    stop = next(e for e in events if e.kind == ExecutorEventKind.STOP)
    assert "usage" in stop.payload
    assert stop.payload["usage"]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_get_bus_returns_non_none(executor):
    bus = executor.get_bus()
    assert bus is not None
    assert hasattr(bus, "push")
    assert hasattr(bus, "wait")


# ── Tool-use round-trip ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_use_round_trip(executor, envelope):
    """
    Full flow: transport yields tool_call → executor yields TOOL_USE
    → test pushes result to bus → executor loops → yields STOP.
    """
    call_num = 0

    async def two_round_stream(api_key, body):
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "function": {"name": "read", "arguments": '{"path":"/foo"}'},
                        }]
                    },
                    "finish_reason": None,
                }],
                "usage": None,
            }
            yield {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0},
            }
        else:
            yield {
                "choices": [{"delta": {"content": "Done."}, "finish_reason": None}],
                "usage": None,
            }
            yield {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 3},
            }

    with patch.object(executor._transport, "stream", side_effect=two_round_stream):
        gen = executor.stream(envelope)

        # 1. Routing TEXT event
        event = await gen.__anext__()
        assert event.kind == ExecutorEventKind.TEXT
        assert "_routing" in event.payload

        # 2. TOOL_USE event
        event = await gen.__anext__()
        assert event.kind == ExecutorEventKind.TOOL_USE
        assert event.payload["tool"] == "read"
        assert event.payload["args"] == {"path": "/foo"}
        tool_id = event.payload["tool_use_id"]

        # Push result BEFORE advancing the generator
        executor.get_bus().push(tool_id, "file contents here")

        # 3. TEXT delta from round 2
        event = await gen.__anext__()
        assert event.kind == ExecutorEventKind.TEXT
        assert event.payload.get("text") == "Done."

        # 4. STOP
        event = await gen.__anext__()
        assert event.kind == ExecutorEventKind.STOP
        assert event.payload["usage"]["output_tokens"] == 3


@pytest.mark.asyncio
async def test_depth_in_routing_event(executor, lineage, policy):
    """Routing event should reflect envelope.depth."""
    from tests.conftest import _build_envelope
    deep_env = _build_envelope(lineage, policy, depth=3)

    async def fake_stream(api_key, body):
        for chunk in _STOP_CHUNKS:
            yield chunk

    with patch.object(executor._transport, "stream", side_effect=fake_stream):
        events = await _collect(executor.stream(deep_env))

    routing = events[0]
    assert routing.payload["_routing"]["depth"] == 3
