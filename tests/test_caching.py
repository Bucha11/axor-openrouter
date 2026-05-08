"""Tests for caching subpackage: breakpoints, ttl_chooser, response_cache."""
from __future__ import annotations

import pytest

from axor_openrouter.caching.breakpoints import apply_cache_control
from axor_openrouter.caching.ttl_chooser import TtlChooser
from axor_openrouter.caching.response_cache import should_cache_response


# ── apply_cache_control ───────────────────────────────────────────────────────

def test_system_string_promoted_to_block_list():
    msgs = [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"}]
    new_msgs, _ = apply_cache_control(msgs, [], ["system"], ttl_seconds=300)
    content = new_msgs[0]["content"]
    assert isinstance(content, list)
    assert content[-1]["cache_control"] == {"type": "ephemeral", "ttl": 300}
    assert content[-1]["text"] == "You are helpful."


def test_system_list_gets_cc_on_last_block():
    msgs = [{"role": "system", "content": [
        {"type": "text", "text": "block1"},
        {"type": "text", "text": "block2"},
    ]}]
    new_msgs, _ = apply_cache_control(msgs, [], ["system"], ttl_seconds=600)
    blocks = new_msgs[0]["content"]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral", "ttl": 600}
    assert blocks[-1]["text"] == "block2"
    assert "cache_control" not in blocks[0]


def test_tools_last_item_gets_cc():
    tools = [{"function": {"name": "read"}}, {"function": {"name": "write"}}]
    _, new_tools = apply_cache_control([], tools, ["tools"], ttl_seconds=3600)
    assert "cache_control" in new_tools[-1]
    assert "cache_control" not in new_tools[0]


def test_empty_blocks_no_change():
    msgs = [{"role": "system", "content": "prompt"}]
    tools = [{"function": {"name": "read"}}]
    new_msgs, new_tools = apply_cache_control(msgs, tools, [], ttl_seconds=300)
    assert new_msgs[0]["content"] == "prompt"
    assert "cache_control" not in new_tools[0]


def test_does_not_mutate_input_lists():
    msgs = [{"role": "system", "content": "prompt"}]
    tools = [{"function": {"name": "read"}}]
    orig_msg, orig_tool = msgs[0], tools[0]
    apply_cache_control(msgs, tools, ["system", "tools"], ttl_seconds=300)
    assert msgs[0] is orig_msg
    assert tools[0] is orig_tool


# ── TtlChooser ────────────────────────────────────────────────────────────────

def test_chooser_system_returns_session_ttl():
    c = TtlChooser(short_ttl="5m", session_ttl="1h")
    assert c.choose("system") == "1h"


def test_chooser_tools_returns_session_ttl():
    c = TtlChooser(short_ttl="5m", session_ttl="1h")
    assert c.choose("tools") == "1h"


def test_chooser_context_top_k_always_short():
    c = TtlChooser(short_ttl="5m", session_ttl="1h")
    assert c.choose("context_top_k") == "5m"


def test_chooser_downgrade_forces_short_for_all():
    c = TtlChooser(short_ttl="5m", session_ttl="1h")
    c.downgrade_to_short()
    assert c.choose("system") == "5m"
    assert c.choose("tools") == "5m"


def test_chooser_restore_lifts_downgrade():
    c = TtlChooser(short_ttl="5m", session_ttl="1h")
    c.downgrade_to_short()
    c.restore_session()
    assert c.choose("system") == "1h"


def test_chooser_record_call_increments_count():
    c = TtlChooser()
    assert c._call_count == 0
    c.record_call()
    c.record_call()
    assert c._call_count == 2


# ── should_cache_response ─────────────────────────────────────────────────────

def test_should_cache_response_false_by_default(envelope):
    assert should_cache_response(envelope) is False


def test_should_cache_response_true_when_deterministic_readonly(lineage, policy):
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.context import ContextView
    from axor_core.contracts.envelope import Capabilities, ExportContract, ExecutionEnvelope
    ctx = ContextView(
        node_id=lineage.node_id, working_summary="",
        visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(["read"]),
        allow_children=False, allow_nested_children=False,
        allow_context_expansion=False, allow_export=True,
        allow_mutation=False, max_child_depth=0,
    )
    env = ExecutionEnvelope(
        node_id=lineage.node_id, task="t", context=ctx, policy=policy,
        capabilities=caps,
        export_contract=ExportContract(mode="summary", allowed_fields=frozenset(), max_export_tokens=None),
        lineage=lineage, cancel_token=make_token(), deterministic=True,
    )
    assert should_cache_response(env) is True


def test_should_cache_response_false_when_write_in_tools(lineage, policy):
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.context import ContextView
    from axor_core.contracts.envelope import Capabilities, ExportContract, ExecutionEnvelope
    ctx = ContextView(
        node_id=lineage.node_id, working_summary="",
        visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(["read", "write"]),
        allow_children=False, allow_nested_children=False,
        allow_context_expansion=False, allow_export=True,
        allow_mutation=True, max_child_depth=0,
    )
    env = ExecutionEnvelope(
        node_id=lineage.node_id, task="t", context=ctx, policy=policy,
        capabilities=caps,
        export_contract=ExportContract(mode="summary", allowed_fields=frozenset(), max_export_tokens=None),
        lineage=lineage, cancel_token=make_token(), deterministic=True,
    )
    assert should_cache_response(env) is False
