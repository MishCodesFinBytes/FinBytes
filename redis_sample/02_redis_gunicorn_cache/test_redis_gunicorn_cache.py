"""
test_redis_gunicorn_cache.py — Tests for redis_gunicorn_cache.py
Uses fakeredis — no live Redis needed.

    pip install fakeredis
    pytest test_redis_gunicorn_cache.py -v
"""

import json
import pytest
import fakeredis

from redis_gunicorn_cache import (
    get_profile,
    get_profile_safe,
    scan_stale_keys,
    delete_stale_keys,
)


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


def test_cache_miss_then_hit(r):
    p1 = get_profile(1, r)
    assert p1["name"] == "Alice"
    p2 = get_profile(1, r)
    assert p2["name"] == "Alice"


def test_ttl_set_on_cache_write(r):
    get_profile(1, r)
    assert r.ttl("user:1") > 0


def test_unknown_user_returns_none(r):
    assert get_profile(999, r) is None


def test_stampede_safe_basic(r):
    profile = get_profile_safe(1, r)
    assert profile["name"] == "Alice"
    profile2 = get_profile_safe(1, r)
    assert profile2["name"] == "Alice"


def test_scan_finds_keys(r):
    get_profile(1, r)
    get_profile(2, r)
    results = scan_stale_keys(r)
    keys = [x["key"] for x in results]
    assert "user:1" in keys
    assert "user:2" in keys


def test_scan_reports_ttl(r):
    get_profile(1, r, ttl=60)
    results = scan_stale_keys(r)
    user1 = next(x for x in results if x["key"] == "user:1")
    assert user1["ttl"] > 0
    assert user1["status"] == "ok"


def test_scan_reports_no_expiry(r):
    r.set("user:99", json.dumps({"name": "NoTTL"}))  # no ex=
    results = scan_stale_keys(r)
    no_ttl = next(x for x in results if x["key"] == "user:99")
    assert no_ttl["status"] == "no-expiry"


def test_delete_stale_keys(r):
    get_profile(1, r)
    get_profile(2, r)
    count = delete_stale_keys(r, "user:*")
    assert count == 2
    assert r.get("user:1") is None
    assert r.get("user:2") is None


def test_rebuild_after_delete(r):
    get_profile(1, r)
    delete_stale_keys(r, "user:*")
    profile = get_profile(1, r)   # re-fetches from DB
    assert profile["name"] == "Alice"
