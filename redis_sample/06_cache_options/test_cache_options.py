"""
test_cache_options.py — Tests for cache_options.py

    pip install fakeredis cachetools
    pytest test_cache_options.py -v
"""

import json
import time
import pytest
import fakeredis

from cache_options import (
    get_with_ttl,
    get_with_lock,
    get_with_ttlcache,
    get_with_background_refresh,
    choose_cache_strategy,
    reset_db_counter,
    db_call_count,
    _ttl_cache,
)


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def reset_counters():
    reset_db_counter()
    _ttl_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: Redis with TTL
# ─────────────────────────────────────────────────────────────────────────────

def test_ttl_cache_miss_fetches_from_db(r):
    result = get_with_ttl("user:1", r)
    assert result["name"] == "Alice"
    assert db_call_count() == 1


def test_ttl_cache_hit_skips_db(r):
    get_with_ttl("user:1", r)
    get_with_ttl("user:1", r)
    assert db_call_count() == 1


def test_ttl_set_after_miss(r):
    get_with_ttl("user:1", r, ttl=60)
    assert r.ttl("user:1") > 0


def test_ttl_unknown_key_returns_none(r):
    result = get_with_ttl("user:missing", r)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: Stampede lock
# ─────────────────────────────────────────────────────────────────────────────

def test_lock_fetches_on_miss(r):
    result = get_with_lock("user:1", r)
    assert result["name"] == "Alice"


def test_lock_hits_cache_on_second_call(r):
    get_with_lock("user:1", r)
    get_with_lock("user:1", r)
    assert db_call_count() == 1


def test_lock_cleans_up_lock_key(r):
    get_with_lock("user:1", r)
    assert r.get("user:1:lock") is None


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: TTLCache (in-process)
# ─────────────────────────────────────────────────────────────────────────────

def test_ttlcache_miss_then_hit():
    _ttl_cache.clear()
    reset_db_counter()
    get_with_ttlcache("user:1")
    get_with_ttlcache("user:1")
    assert db_call_count() == 1


def test_ttlcache_returns_correct_data():
    _ttl_cache.clear()
    result = get_with_ttlcache("user:2")
    assert result["name"] == "Bob"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: Background refresh
# ─────────────────────────────────────────────────────────────────────────────

def test_background_refresh_serves_cached_immediately(r):
    r.set("user:1", json.dumps({"name": "Alice", "balance": 1500.0}), ex=200)
    result = get_with_background_refresh("user:1", r)
    assert result["name"] == "Alice"


def test_background_refresh_triggers_on_low_ttl(r):
    # Set key with very low TTL to trigger background refresh
    r.set("user:1", json.dumps({"name": "Alice", "balance": 1500.0}), ex=10)
    reset_db_counter()
    get_with_background_refresh("user:1", r, refresh_threshold=30)
    time.sleep(0.1)  # allow background thread to run
    assert db_call_count() >= 1


def test_background_refresh_cold_start(r):
    result = get_with_background_refresh("user:1", r)
    assert result is not None
    assert result["name"] == "Alice"


# ─────────────────────────────────────────────────────────────────────────────
# Strategy selector
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_pure_computation():
    assert choose_cache_strategy(False, False, True, False) == "lru_cache"


def test_strategy_single_server_no_invalidation():
    assert choose_cache_strategy(False, False, False, False) == "TTLCache (cachetools)"


def test_strategy_high_traffic_multi_server():
    assert choose_cache_strategy(True, True, False, True) == "Redis + stampede lock"


def test_strategy_standard_multi_server():
    assert choose_cache_strategy(True, True, False, False) == "Redis with TTL"
