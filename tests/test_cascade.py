"""Tests for cascade: TierMapper depth → model resolution."""
from __future__ import annotations

import pytest

from axor_openrouter.cascade.tiers import TierMapper, TierSpec, DEFAULT_TIERS


@pytest.fixture()
def mapper():
    return TierMapper(DEFAULT_TIERS)


def test_depth_0_returns_opus(mapper):
    assert mapper.resolve(0) == "anthropic/claude-opus-4-7"

def test_depth_1_returns_sonnet(mapper):
    assert mapper.resolve(1) == "anthropic/claude-sonnet-4-6"

def test_depth_2_returns_sonnet(mapper):
    assert mapper.resolve(2) == "anthropic/claude-sonnet-4-6"

def test_depth_3_returns_gpt4o_mini(mapper):
    assert mapper.resolve(3) == "openai/gpt-4o-mini"

def test_depth_5_returns_gpt4o_mini(mapper):
    assert mapper.resolve(5) == "openai/gpt-4o-mini"

def test_depth_6_returns_llama(mapper):
    assert mapper.resolve(6) == "meta-llama/llama-3.3-70b-instruct"

def test_depth_100_returns_llama_last_tier(mapper):
    assert mapper.resolve(100) == "meta-llama/llama-3.3-70b-instruct"

def test_tier_shift_plus1_upgrades_to_more_capable(mapper):
    # depth=1 → sonnet (tier_index=1); shift +1 → opus (tier_index=0)
    model = mapper.resolve(1, tier_shift=1)
    assert model == "anthropic/claude-opus-4-7"

def test_tier_shift_minus1_downgrades_to_cheaper(mapper):
    # depth=1 → sonnet (tier_index=1); shift -1 → gpt-4o-mini (tier_index=2)
    model = mapper.resolve(1, tier_shift=-1)
    assert model == "openai/gpt-4o-mini"

def test_tier_shift_zero_no_change(mapper):
    assert mapper.resolve(0, tier_shift=0) == "anthropic/claude-opus-4-7"

def test_tier_shift_clamped_at_most_capable(mapper):
    # Already at tier_index=0 (opus); shift +10 stays at opus
    assert mapper.resolve(0, tier_shift=10) == "anthropic/claude-opus-4-7"

def test_tier_shift_clamped_at_cheapest(mapper):
    # Already at tier_index=3 (llama); shift -10 stays at llama
    assert mapper.resolve(6, tier_shift=-10) == "meta-llama/llama-3.3-70b-instruct"

def test_tier_label_returns_string(mapper):
    assert mapper.tier_label(0) == "tier_0"
    assert mapper.tier_label(1) == "tier_1"
    assert mapper.tier_label(6) == "tier_3"

def test_custom_tier_spec_matches():
    spec = TierSpec(min_depth=0, max_depth=2, model="custom/model", tier_index=0)
    assert spec.matches(0)
    assert spec.matches(2)
    assert not spec.matches(3)

def test_open_ended_tier_spec_matches_any_depth():
    spec = TierSpec(min_depth=5, max_depth=None, model="x", tier_index=0)
    assert spec.matches(5)
    assert spec.matches(100)
    assert not spec.matches(4)
