"""Tests for routing subpackage: provider_prefs, fallbacks, byok."""
from __future__ import annotations

import pytest

from axor_openrouter.routing.provider_prefs import ProviderPrefs
from axor_openrouter.routing.fallbacks import FallbackChain, DEFAULT_FALLBACKS, get_fallbacks
from axor_openrouter.routing.byok import BYOKConfig


# ── ProviderPrefs ─────────────────────────────────────────────────────────────

def test_empty_prefs_give_empty_dict():
    assert ProviderPrefs().to_dict() == {}

def test_sort_included():
    d = ProviderPrefs(sort="throughput").to_dict()
    assert d["sort"] == "throughput"

def test_no_sort_not_in_dict():
    d = ProviderPrefs().to_dict()
    assert "sort" not in d

def test_max_prompt_price():
    d = ProviderPrefs(max_prompt_price=5_000_000).to_dict()
    assert "max_price" in d
    assert "prompt" in d["max_price"]

def test_allow_fallbacks_false_included():
    d = ProviderPrefs(allow_fallbacks=False).to_dict()
    assert d["allow_fallbacks"] is False

def test_allow_fallbacks_true_omitted():
    d = ProviderPrefs(allow_fallbacks=True).to_dict()
    assert "allow_fallbacks" not in d

def test_order_list():
    d = ProviderPrefs(order=["Anthropic", "OpenAI"]).to_dict()
    assert d["order"] == ["Anthropic", "OpenAI"]

def test_byok_injected():
    d = ProviderPrefs(byok_provider="openai", byok_api_key="sk-x").to_dict()
    assert d["api_key_or_org_ids"]["openai"] == "sk-x"

def test_quantizations():
    d = ProviderPrefs(quantizations=["fp16"]).to_dict()
    assert d["quantizations"] == ["fp16"]


# ── FallbackChain ─────────────────────────────────────────────────────────────

def test_fallback_chain_stores_models():
    chain = FallbackChain(tier=0, models=["a", "b"])
    assert chain.models == ["a", "b"]

def test_fallback_chain_is_empty():
    assert FallbackChain(tier=0, models=[]).is_empty()

def test_fallback_chain_not_empty():
    assert not FallbackChain(tier=0, models=["m"]).is_empty()

def test_default_fallbacks_has_tier_0():
    assert 0 in DEFAULT_FALLBACKS
    assert len(DEFAULT_FALLBACKS[0]) > 0

def test_get_fallbacks_known_tier_returns_list():
    result = get_fallbacks(0)
    assert isinstance(result, list)
    assert len(result) > 0

def test_get_fallbacks_unknown_tier_returns_fallback_default():
    result = get_fallbacks(99)
    assert isinstance(result, list)
    assert len(result) > 0


# ── BYOKConfig ────────────────────────────────────────────────────────────────

def test_byok_to_provider_dict():
    cfg = BYOKConfig(provider="anthropic", api_key="sk-ant-test")
    d = cfg.to_provider_dict()
    assert d["api_key_or_org_ids"]["anthropic"] == "sk-ant-test"
