"""Tests for tools.py schema helpers."""
from __future__ import annotations

from axor_openrouter.tools import build_tools, TOOL_SCHEMAS


def test_all_schemas_have_required_keys():
    for name, schema in TOOL_SCHEMAS.items():
        assert schema["type"] == "function", f"{name} missing type=function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn


def test_build_tools_returns_list(envelope):
    tools = build_tools(envelope)
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_build_tools_no_duplicates(envelope):
    tools = build_tools(envelope)
    names = [t["function"]["name"] for t in tools]
    assert len(names) == len(set(names))


def test_each_tool_has_function_wrapper(envelope):
    tools = build_tools(envelope)
    for t in tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
