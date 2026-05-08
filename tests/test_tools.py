"""Unit tests for tools.py schema helpers."""
from __future__ import annotations

import pytest

from axor_core.contracts.envelope import ExecutionEnvelope
from axor_openrouter.tools import build_tools, TOOL_SCHEMAS


@pytest.fixture()
def envelope():
    return ExecutionEnvelope(task="task", context_text="")


def test_all_schemas_have_required_keys():
    for name, schema in TOOL_SCHEMAS.items():
        assert schema["type"] == "function", f"{name} missing type=function"
        fn = schema["function"]
        assert "name" in fn, f"{name} missing function.name"
        assert "description" in fn, f"{name} missing function.description"
        assert "parameters" in fn, f"{name} missing function.parameters"


def test_build_tools_returns_list(envelope):
    tools = build_tools(envelope)
    assert isinstance(tools, list)
    assert len(tools) > 0


def test_build_tools_no_duplicates(envelope):
    tools = build_tools(envelope)
    names = [t["function"]["name"] for t in tools]
    assert len(names) == len(set(names)), "Duplicate tool names detected"
