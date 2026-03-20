"""
test_troubleshooting_redis.py — Tests for troubleshooting_redis.py

    pip install fakeredis
    pytest test_troubleshooting_redis.py -v
"""

import pytest
import fakeredis

from troubleshooting_redis import (
    demo_cache_miss,
    demo_ttl_expiry,
    demo_stale_data,
    fix_stale_data,
    demo_redis_down,
    demo_stampede_protection,
    demo_flush_recovery,
    run_debug_checklist,
    read_through,
    _db,
)


@pytest.fixture
def r():
    redis = fakeredis.FakeRedis(decode_responses=True)
    yield redis
    redis.flushall()


def test_cache_miss_rebuilds(r):
    result = demo_cache_miss(r)
    assert result["miss_then_hit"] is True


def test_ttl_expiry_rebuilds(r):
    result = demo_ttl_expiry(r)
    assert result["rebuilt_after_expiry"] is True


def test_stale_data_detected(r):
    _db["user:1"]["balance"] = 1000.00  # reset
    result = demo_stale_data(r)
    assert result["stale_detected"] is True


def test_stale_data_fix_returns_fresh(r):
    _db["user:1"]["balance"] = 500.00
    result = fix_stale_data(r)
    assert result["fresh_balance"] == 500.00


def test_redis_down_fallback_survives():
    result = demo_redis_down()
    assert result["survived"] is True
    assert result["name"] == "Alice"


def test_stampede_protection_one_db_call(r):
    result = demo_stampede_protection(r)
    assert result["db_calls"] == 1
    assert result["all_got_result"] is True


def test_flush_recovery(r):
    result = demo_flush_recovery(r)
    assert result["data_survived"] is True
    assert result["cache_rebuilt"] is True


def test_debug_checklist_all_green(r):
    checks = run_debug_checklist(r)
    assert checks["redis_up"] is True
    assert checks["keys_expiring"] is True
    assert checks["fallback_working"] is True
    assert checks["db_healthy"] is True


def test_read_through_returns_none_for_unknown(r):
    result = read_through("user:999", r)
    assert result is None
