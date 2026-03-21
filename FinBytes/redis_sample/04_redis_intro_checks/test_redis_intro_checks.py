"""
test_redis_intro_checks.py — Tests for redis_intro_checks.py
Uses fakeredis — no live Redis needed.

    pip install fakeredis
    pytest test_redis_intro_checks.py -v
"""

import json
import pytest
import fakeredis

from redis_intro_checks import (
    check_connectivity,
    check_key,
    get_user_profile,
    get_user_balance,
    format_currency,
    simulate_flush_and_rebuild,
)


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


def test_connectivity_returns_true(r):
    assert check_connectivity(r) is True


def test_cache_miss_fetches_profile(r):
    profile = get_user_profile(1, r)
    assert profile["name"] == "Alice"


def test_cache_hit_on_second_call(r):
    get_user_profile(1, r)
    r.set("user:1", json.dumps({"name": "CachedAlice", "balance": 0}))
    profile = get_user_profile(1, r)
    assert profile["name"] == "CachedAlice"


def test_ttl_set_after_miss(r):
    get_user_profile(2, r)
    assert r.ttl("user:2") > 0


def test_unknown_user_returns_none(r):
    assert get_user_profile(999, r) is None


def test_format_currency_pure():
    assert format_currency(1500.0) == "£1,500.00"
    assert format_currency(275.5) == "£275.50"


def test_get_user_balance_reads_redis(r):
    r.set("balance:1", "1500.00")
    result = get_user_balance(1, r)
    assert result == "£1,500.00"


def test_get_user_balance_zero_when_missing(r):
    result = get_user_balance(99, r)
    assert result == "£0.00"


def test_flush_and_rebuild_repopulates_cache(r):
    get_user_profile(1, r)
    get_user_profile(2, r)
    simulate_flush_and_rebuild(r)
    # After rebuild, keys should be back in cache
    assert r.get("user:1") is not None
    assert r.get("user:2") is not None


def test_key_inspection_does_not_raise(r, capsys):
    r.set("user:1", json.dumps({"name": "Alice"}), ex=60)
    check_key(r, "user:1")     # should log without raising
    check_key(r, "missing:key")
