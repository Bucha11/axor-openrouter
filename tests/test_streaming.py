"""Tests for StreamAccumulator."""
from __future__ import annotations

from axor_openrouter.transport import StreamAccumulator


def test_accumulate_text_chunks():
    acc = StreamAccumulator()
    acc.feed({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}], "usage": None})
    acc.feed({"choices": [{"delta": {"content": " world"}, "finish_reason": None}], "usage": None})
    acc.feed({"choices": [{"delta": {}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    assert acc.text == "Hello world"
    assert acc.finish_reason == "stop"
    assert acc.usage["prompt_tokens"] == 10


def test_accumulate_tool_call_arguments():
    acc = StreamAccumulator()
    acc.feed({
        "choices": [{
            "delta": {
                "tool_calls": [{"index": 0, "id": "call_abc",
                                 "function": {"name": "read", "arguments": '{"path":'}}]
            },
            "finish_reason": None,
        }],
        "usage": None,
    })
    acc.feed({
        "choices": [{
            "delta": {
                "tool_calls": [{"index": 0, "function": {"arguments": ' "/foo"}'}}]
            },
            "finish_reason": None,
        }],
        "usage": None,
    })
    acc.feed({
        "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    })

    assert acc.finish_reason == "tool_calls"
    assert len(acc.tool_calls) == 1
    tc = acc.tool_calls[0]
    assert tc["function"]["name"] == "read"
    assert tc["function"]["arguments"] == '{"path": "/foo"}'


def test_no_usage_until_final():
    acc = StreamAccumulator()
    acc.feed({"choices": [{"delta": {"content": "x"}, "finish_reason": None}], "usage": None})
    assert acc.usage == {}


def test_has_tool_calls_false_initially():
    acc = StreamAccumulator()
    assert acc.has_tool_calls() is False


def test_has_tool_calls_true_after_tool_delta():
    acc = StreamAccumulator()
    acc.feed({
        "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                                "function": {"name": "bash", "arguments": ""}}]},
                     "finish_reason": None}],
        "usage": None,
    })
    assert acc.has_tool_calls() is True


def test_multiple_tool_calls_in_order():
    acc = StreamAccumulator()
    for i in range(3):
        acc.feed({
            "choices": [{"delta": {"tool_calls": [{"index": i, "id": f"c{i}",
                                                    "function": {"name": f"tool{i}",
                                                                 "arguments": "{}"}}]},
                         "finish_reason": None}],
            "usage": None,
        })
    acc.feed({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
              "usage": {"prompt_tokens": 5, "completion_tokens": 0}})
    calls = acc.tool_calls
    assert len(calls) == 3
    assert calls[0]["function"]["name"] == "tool0"
    assert calls[2]["function"]["name"] == "tool2"
