"""Tests for governance subpackage: AdaptiveRouter, CacheHealthMonitor."""
from __future__ import annotations

import pytest

from axor_openrouter.governance.adaptive_router import AdaptiveRouter
from axor_openrouter.governance.cache_health import CacheHealthMonitor
from axor_openrouter.caching.ttl_chooser import TtlChooser


# ── AdaptiveRouter ──────────────────────────────────────────────────────

def test_initial_shift_is_zero():
    r = AdaptiveRouter()
    assert r.current_shift() == 0

def test_apply_shift_increments():
    r = AdaptiveRouter()
    r.apply_shift(1)
    assert r.current_shift() == 1

def test_apply_shift_accumulates():
    r = AdaptiveRouter()
    r.apply_shift(1)
    r.apply_shift(1)
    assert r.current_shift() == 2

def test_apply_shift_clamped_at_max():
    r = AdaptiveRouter(max_shift=2)
    r.apply_shift(5)
    assert r.current_shift() == 2

def test_apply_shift_clamped_at_min():
    r = AdaptiveRouter(min_shift=-2)
    r.apply_shift(-5)
    assert r.current_shift() == -2

def test_apply_shift_negative():
    r = AdaptiveRouter()
    r.apply_shift(2)
    r.apply_shift(-1)
    assert r.current_shift() == 1

def test_reset_returns_to_zero():
    r = AdaptiveRouter()
    r.apply_shift(2)
    r.reset()
    assert r.current_shift() == 0

def test_set_shift_is_absolute():
    r = AdaptiveRouter()
    r.set_shift(-1)
    r.set_shift(-1)  # calling twice stays at -1, not -2
    assert r.current_shift() == -1

def test_set_shift_clamped_at_max():
    r = AdaptiveRouter(max_shift=1)
    r.set_shift(5)
    assert r.current_shift() == 1

def test_set_shift_clamped_at_min():
    r = AdaptiveRouter(min_shift=-1)
    r.set_shift(-5)
    assert r.current_shift() == -1

def test_set_shift_overrides_previous_apply():
    r = AdaptiveRouter()
    r.apply_shift(2)
    r.set_shift(0)
    assert r.current_shift() == 0


# ── CacheHealthMonitor ────────────────────────────────────────────────

@pytest.fixture()
def chooser():
    return TtlChooser(short_ttl="5m", session_ttl="1h")

@pytest.fixture()
def monitor(chooser):
    return CacheHealthMonitor(ttl_chooser=chooser, window=10, threshold=0.5)


def test_initial_hit_rate_is_one(monitor):
    assert monitor.hit_rate() == 1.0

def test_all_hits_rate_is_one(monitor):
    for _ in range(5):
        monitor.record_hit()
    assert monitor.hit_rate() == 1.0

def test_all_misses_rate_is_zero(monitor):
    for _ in range(5):
        monitor.record_miss()
    assert monitor.hit_rate() == 0.0

def test_mixed_rate(monitor):
    monitor.record_hit()
    monitor.record_miss()
    assert monitor.hit_rate() == 0.5

def test_downgrade_triggered_below_threshold(monitor, chooser):
    for _ in range(6):
        monitor.record_miss()
    assert monitor._downgraded is True
    assert chooser.choose("system") == "5m"

def test_no_downgrade_above_threshold(monitor, chooser):
    for _ in range(6):
        monitor.record_hit()
    assert monitor._downgraded is False
    assert chooser.choose("system") == "1h"

def test_recovery_restores_session_ttl(monitor, chooser):
    for _ in range(6):
        monitor.record_miss()
    assert monitor._downgraded is True
    for _ in range(10):
        monitor.record_hit()
    assert monitor._downgraded is False
    assert chooser.choose("system") == "1h"

def test_record_tool_call_is_noop(monitor):
    monitor.record_tool_call()
    assert monitor.hit_rate() == 1.0

def test_record_usage_hit_when_cache_read(monitor):
    monitor.record_usage(cache_read=500, total_input=1000)
    assert len(monitor._hits) == 1
    assert monitor._hits[0] is True

def test_record_usage_miss_when_no_cache_read(monitor):
    monitor.record_usage(cache_read=0, total_input=1000)
    assert monitor._hits[0] is False
