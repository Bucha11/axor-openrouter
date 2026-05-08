"""Tests for usage_codec.decode_usage."""
from __future__ import annotations

from axor_openrouter.usage_codec import decode_usage


def test_basic_mapping():
    u = decode_usage({"prompt_tokens": 100, "completion_tokens": 50})
    assert u.input_tokens == 100
    assert u.output_tokens == 50
    assert u.cache_read_input_tokens == 0
    assert u.cache_creation_input_tokens == 0


def test_cache_read_subtracted():
    u = decode_usage({"prompt_tokens": 1000, "completion_tokens": 20,
                      "cache_read_input_tokens": 900})
    assert u.cache_read_input_tokens == 900
    assert u.input_tokens == 100


def test_cache_write_subtracted():
    u = decode_usage({"prompt_tokens": 500, "completion_tokens": 10,
                      "cache_write_tokens": 400})
    assert u.cache_creation_input_tokens == 400
    assert u.input_tokens == 100


def test_both_cache_fields():
    u = decode_usage({"prompt_tokens": 2000, "completion_tokens": 30,
                      "cache_read_input_tokens": 1500,
                      "cache_write_tokens": 300})
    assert u.cache_read_input_tokens == 1500
    assert u.cache_creation_input_tokens == 300
    assert u.input_tokens == 200


def test_uncached_input_never_negative():
    u = decode_usage({"prompt_tokens": 100, "completion_tokens": 5,
                      "cache_read_input_tokens": 200})
    assert u.input_tokens == 0


def test_context_tokens_forwarded():
    u = decode_usage({"prompt_tokens": 50, "completion_tokens": 10}, context_tokens=500)
    assert u.context_tokens == 500


def test_empty_dict():
    u = decode_usage({})
    assert u.input_tokens == 0
    assert u.output_tokens == 0


def test_alias_input_cache_read_tokens():
    u = decode_usage({"prompt_tokens": 200, "completion_tokens": 5,
                      "input_cache_read_tokens": 150})
    assert u.cache_read_input_tokens == 150


def test_alias_prompt_cache_hit_tokens():
    u = decode_usage({"prompt_tokens": 300, "completion_tokens": 10,
                      "prompt_cache_hit_tokens": 250})
    assert u.cache_read_input_tokens == 250


def test_alias_prompt_cache_miss_tokens():
    u = decode_usage({"prompt_tokens": 400, "completion_tokens": 5,
                      "prompt_cache_miss_tokens": 350})
    assert u.cache_creation_input_tokens == 350
