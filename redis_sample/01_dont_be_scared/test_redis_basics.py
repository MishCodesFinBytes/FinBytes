"""
test_redis_basics.py — Tests for redis_basics.py
No live Redis required — uses fakeredis for isolation.

Run with:
    pip install fakeredis
    pytest test_redis_basics.py -v
"""

import json
import pytest
import fakeredis

from redis_basics import (
    get_user_profile,
    invalidate_user,
    format_currency,
    check_redis_health,
)


@pytest.fixture
def r():
    """Fake in-memory Redis — no server needed."""
    return fakeredis.FakeRedis(decode_responses=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core cache behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_miss_fetches_from_db(r):
    profile = get_user_profile(1, r)
    assert profile is not None
    assert profile["name"] == "Alice"


def test_cache_hit_on_second_call(r):
    get_user_profile(1, r)           # populates cache
    # Overwrite the DB value to prove we're reading from Redis
    r.set("user:1", json.dumps({"name": "Cached Alice", "balance": 0}))
    profile = get_user_profile(1, r)
    assert profile["name"] == "Cached Alice"


def test_ttl_is_set_after_cache_miss(r):
    get_user_profile(2, r)
    ttl = r.ttl("user:2")
    assert ttl > 0
    assert ttl <= 300


def test_unknown_user_returns_none(r):
    profile = get_user_profile(999, r)
    assert profile is None


def test_none_not_cached_for_unknown_user(r):
    get_user_profile(999, r)
    assert r.get("user:999") is None


# ─────────────────────────────────────────────────────────────────────────────
# Invalidation
# ─────────────────────────────────────────────────────────────────────────────

def test_invalidate_removes_key(r):
    get_user_profile(1, r)
    assert r.get("user:1") is not None
    invalidate_user(1, r)
    assert r.get("user:1") is None


def test_access_after_invalidation_hits_db(r):
    get_user_profile(1, r)
    invalidate_user(1, r)
    profile = get_user_profile(1, r)   # should re-fetch from DB
    assert profile["name"] == "Alice"


# ─────────────────────────────────────────────────────────────────────────────
# Pure function (lru_cache) — no Redis involved
# ─────────────────────────────────────────────────────────────────────────────

def test_format_currency_gbp():
    assert format_currency(1500.0) == "£1,500.00"
    assert format_currency(275.5) == "£275.50"


def test_format_currency_is_deterministic():
    assert format_currency(100.0) == format_currency(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

def test_health_check_returns_true_when_alive(r):
    assert check_redis_health(r) is True


def test_health_check_returns_false_when_down(monkeypatch):
    import redis as redis_lib
    bad_r = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(bad_r, "ping", lambda: (_ for _ in ()).throw(
        redis_lib.exceptions.ConnectionError("down")
    ))
    assert check_redis_health(bad_r) is False
