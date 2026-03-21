"""
test_lru_cache_betrayal.py — Tests for lru_cache_betrayal.py
Self-contained — uses fakeredis internally.

    pip install fakeredis
    pytest test_lru_cache_betrayal.py -v
"""

import pytest
from lru_cache_betrayal import (
    format_currency,
    get_formatted_balance,
    Worker,
    demonstrate_inconsistency,
    compile_pattern,
    build_sql_template,
    _set_balance,
    _shared_redis,
)


@pytest.fixture(autouse=True)
def clean_redis():
    """Reset the shared fake Redis between tests."""
    _shared_redis.flushall()
    yield
    _shared_redis.flushall()


# ─────────────────────────────────────────────────────────────────────────────
# Pure function: lru_cache is safe
# ─────────────────────────────────────────────────────────────────────────────

def test_format_currency_correct():
    assert format_currency(1500.0) == "£1,500.00"
    assert format_currency(0.5) == "£0.50"


def test_format_currency_deterministic():
    assert format_currency(100.0) == format_currency(100.0)


def test_get_formatted_balance_reads_redis():
    _set_balance(1, 1500.00)
    result = get_formatted_balance(1)
    assert result == "£1,500.00"


def test_get_formatted_balance_zero_when_missing():
    result = get_formatted_balance(99)
    assert result == "£0.00"


# ─────────────────────────────────────────────────────────────────────────────
# Worker with local cache — demonstrates the inconsistency bug
# ─────────────────────────────────────────────────────────────────────────────

def test_worker_local_cache_serves_stale_value():
    """After Redis is updated, worker with local cache still serves old value."""
    _set_balance(1, 1000.00)
    worker = Worker("TestWorkerA")

    first = worker.get_balance_with_local_cache(1)
    assert first == "1000.0"

    # Update Redis — worker's local cache is NOT invalidated
    _set_balance(1, 750.00)

    stale = worker.get_balance_with_local_cache(1)
    assert stale == "1000.0"   # still the old value — the bug


def test_worker_redis_only_always_fresh():
    """Worker reading directly from Redis always gets the current value."""
    _set_balance(1, 1000.00)
    worker = Worker("TestWorkerB")

    worker.get_balance_redis_only(1)
    _set_balance(1, 750.00)

    fresh = worker.get_balance_redis_only(1)
    assert fresh == "750.0"   # correct, always current


def test_inconsistency_between_workers():
    """Demonstrates that local-cache worker and Redis-only worker disagree."""
    result = demonstrate_inconsistency()
    assert result["inconsistent"] is True
    assert result["worker_a"] != result["worker_b"]


# ─────────────────────────────────────────────────────────────────────────────
# Safe lru_cache uses
# ─────────────────────────────────────────────────────────────────────────────

def test_compile_pattern_returns_regex():
    pattern = compile_pattern(r"\d{4}")
    assert pattern.match("2024")
    assert not pattern.match("abc")


def test_build_sql_template_correct():
    sql = build_sql_template("users")
    assert "users" in sql
    assert "SELECT" in sql


def test_build_sql_template_deterministic():
    assert build_sql_template("orders") == build_sql_template("orders")
