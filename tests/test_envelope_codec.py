"""Tests for envelope_codec: build_messages, append_tool_result, _ttl_to_seconds."""
from __future__ import annotations

import pytest

from axor_openrouter.envelope_codec import (
    _ttl_to_seconds,
    append_tool_result,
    build_messages,
    build_tool_request_messages,
)


# ── _ttl_to_seconds ───────────────────────────────────────────────────────────

def test_ttl_minutes():
    assert _ttl_to_seconds("5m") == 300

def test_ttl_hours():
    assert _ttl_to_seconds("1h") == 3600

def test_ttl_seconds_suffix():
    assert _ttl_to_seconds("30s") == 30

def test_ttl_raw_integer_string():
    assert _ttl_to_seconds("120") == 120

def test_ttl_fallback_on_invalid():
    assert _ttl_to_seconds("???") == 300


# ── build_messages ────────────────────────────────────────────────────────────

def test_minimal_envelope_produces_single_user_message(envelope):
    msgs = build_messages(envelope)
    assert len(msgs) == 1
    assert msgs[0] == {"role": "user", "content": "say hello"}


def test_skill_fragment_produces_system_message(envelope_with_skill):
    msgs = build_messages(envelope_with_skill)
    assert msgs[0]["role"] == "system"
    assert "Be helpful" in msgs[0]["content"]
    assert msgs[-1]["role"] == "user"


def test_no_system_message_without_skill_fragments(envelope):
    msgs = build_messages(envelope)
    assert all(m["role"] != "system" for m in msgs)


def test_cache_hints_system_adds_block_list(envelope_with_skill):
    msgs = build_messages(
        envelope_with_skill,
        cache_hints={"blocks": ["system"], "ttl": "5m"},
    )
    system_content = msgs[0]["content"]
    assert isinstance(system_content, list)
    cc = system_content[-1]["cache_control"]
    assert cc["type"] == "ephemeral"
    assert cc["ttl"] == 300


def test_cache_hints_system_1h_ttl(envelope_with_skill):
    msgs = build_messages(
        envelope_with_skill,
        cache_hints={"blocks": ["system"], "ttl": "1h"},
    )
    assert msgs[0]["content"][-1]["cache_control"]["ttl"] == 3600


def test_cache_hints_from_envelope_used_when_no_override(envelope_with_skill):
    envelope_with_skill.cache_hints = {"blocks": ["system"], "ttl": "5m"}
    msgs = build_messages(envelope_with_skill)
    assert isinstance(msgs[0]["content"], list)


# ── append_tool_result ────────────────────────────────────────────────────────

def test_append_tool_result_string(envelope):
    msgs = build_messages(envelope)
    append_tool_result(msgs, "call_1", "file content")
    last = msgs[-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "call_1"
    assert last["content"] == "file content"


def test_append_tool_result_dict_json(envelope):
    msgs = build_messages(envelope)
    append_tool_result(msgs, "call_2", {"key": "val"})
    assert '"key"' in msgs[-1]["content"]


def test_append_tool_result_denied(envelope):
    msgs = build_messages(envelope)
    append_tool_result(msgs, "call_3", {"error": "tool_denied", "reason": "not allowed"})
    assert "Tool denied" in msgs[-1]["content"]


def test_append_tool_result_integer(envelope):
    msgs = build_messages(envelope)
    append_tool_result(msgs, "call_4", 42)
    assert msgs[-1]["content"] == "42"


# ── build_tool_request_messages ───────────────────────────────────────────────

def test_build_tool_request_messages_appends_assistant_turn(envelope):
    msgs = build_messages(envelope)
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
    new_msgs = build_tool_request_messages(msgs, None, tool_calls)
    assert len(new_msgs) == len(msgs) + 1
    last = new_msgs[-1]
    assert last["role"] == "assistant"
    assert last["tool_calls"] == tool_calls


def test_build_tool_request_messages_does_not_mutate_original(envelope):
    msgs = build_messages(envelope)
    orig_len = len(msgs)
    build_tool_request_messages(msgs, None, [])
    assert len(msgs) == orig_len
