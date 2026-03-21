"""
test_request_tracer.py — Tests for request_tracer.py
Uses fakeredis — no live Redis needed.

    pip install fakeredis
    pytest test_request_tracer.py -v
"""

import json
import pytest
import fakeredis

from request_tracer import (
    handle_request,
    browser_sends_request,
    redis_layer,
    database_layer,
)


@pytest.fixture
def r():
    return fakeredis.FakeRedis(decode_responses=True)


def test_full_request_cache_miss_returns_profile(r):
    response = handle_request(1, r)
    assert response["profile"] is not None
    assert response["profile"]["name"] == "Alice"


def test_full_request_cache_hit_on_second_call(r):
    handle_request(1, r)
    response = handle_request(1, r)
    assert response["profile"]["name"] == "Alice"


def test_response_includes_request_id(r):
    response = handle_request(1, r)
    assert "request_id" in response
    assert len(response["request_id"]) == 8


def test_response_includes_elapsed(r):
    response = handle_request(1, r)
    assert "elapsed_ms" in response


def test_cache_populated_after_miss(r):
    handle_request(1, r)
    assert r.get("user:1") is not None


def test_ttl_set_after_db_fetch(r):
    handle_request(1, r)
    assert r.ttl("user:1") > 0


def test_unknown_user_returns_none_profile(r):
    response = handle_request(999, r)
    assert response["profile"] is None


def test_unknown_user_not_cached(r):
    handle_request(999, r)
    assert r.get("user:999") is None


def test_redis_layer_cache_hit(r):
    r.set("user:2", json.dumps({"user_id": 2, "name": "Bob", "balance": 275.5}))
    ctx = browser_sends_request(2)
    profile = redis_layer(ctx, r)
    assert profile["name"] == "Bob"


def test_correlation_id_is_unique():
    ctx1 = browser_sends_request(1)
    ctx2 = browser_sends_request(1)
    assert ctx1.request_id != ctx2.request_id
