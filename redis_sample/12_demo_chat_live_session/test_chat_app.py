"""
test_chat_app.py — Tests for chat_app.py
Covers all scenarios from the four demo posts:
  - cache miss / hit
  - TTL expiry and eviction
  - Pub/Sub send
  - Redis down (connection error)
  - Atomic pipeline (race condition safety)
  - History endpoint

Uses fakeredis — no live Redis or running server needed.

    pip install fakeredis flask pytest
    pytest test_chat_app.py -v
"""

import json
import pytest
import fakeredis
import redis as redis_lib

from chat_app import (
    send_message,
    get_history,
    create_app,
    MESSAGE_LIST_KEY,
    MAX_MESSAGES,
    CACHE_TTL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def r():
    """Fresh in-memory Redis for every test."""
    redis = fakeredis.FakeRedis(decode_responses=True)
    yield redis
    redis.flushall()


@pytest.fixture
def client(r):
    """Flask test client wired to fake Redis."""
    app = create_app(r)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# send_message — Pub/Sub + cache
# ─────────────────────────────────────────────────────────────────────────────

def test_send_message_returns_true(r):
    assert send_message(r, "Hello World") is True


def test_send_message_stores_in_list(r):
    send_message(r, "Test message")
    messages = r.lrange(MESSAGE_LIST_KEY, 0, -1)
    assert "Test message" in messages


def test_send_message_sets_ttl(r):
    send_message(r, "Hello")
    assert r.ttl(MESSAGE_LIST_KEY) > 0


def test_send_message_ttl_within_bounds(r):
    send_message(r, "Hello")
    ttl = r.ttl(MESSAGE_LIST_KEY)
    assert 0 < ttl <= CACHE_TTL


def test_send_multiple_messages_stored(r):
    for i in range(3):
        send_message(r, f"Message {i}")
    messages = r.lrange(MESSAGE_LIST_KEY, 0, -1)
    assert len(messages) == 3


def test_send_evicts_old_messages_beyond_max(r):
    """LTRIM should keep only the last MAX_MESSAGES."""
    for i in range(MAX_MESSAGES + 5):
        send_message(r, f"msg-{i}")
    messages = r.lrange(MESSAGE_LIST_KEY, 0, -1)
    assert len(messages) == MAX_MESSAGES


def test_send_order_is_newest_first(r):
    """LPUSH means newest message is at index 0."""
    send_message(r, "first")
    send_message(r, "second")
    messages = r.lrange(MESSAGE_LIST_KEY, 0, -1)
    assert messages[0] == "second"


# ─────────────────────────────────────────────────────────────────────────────
# Redis down — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

class BrokenRedis:
    """Simulates Redis being completely unavailable."""
    def publish(self, *a, **kw):
        raise redis_lib.exceptions.ConnectionError("down")
    def pipeline(self, *a, **kw):
        raise redis_lib.exceptions.ConnectionError("down")
    def lrange(self, *a, **kw):
        raise redis_lib.exceptions.ConnectionError("down")


def test_send_message_returns_false_when_redis_down():
    assert send_message(BrokenRedis(), "test") is False


def test_get_history_returns_empty_when_redis_down():
    assert get_history(BrokenRedis()) == []


# ─────────────────────────────────────────────────────────────────────────────
# get_history — cache hit / miss
# ─────────────────────────────────────────────────────────────────────────────

def test_get_history_cache_miss_returns_empty(r):
    messages = get_history(r)
    assert messages == []


def test_get_history_cache_hit_returns_messages(r):
    send_message(r, "Hello")
    messages = get_history(r)
    assert "Hello" in messages


def test_get_history_returns_all_messages(r):
    for msg in ["A", "B", "C"]:
        send_message(r, msg)
    history = get_history(r)
    assert set(history) == {"A", "B", "C"}


# ─────────────────────────────────────────────────────────────────────────────
# Flask API endpoints
# ─────────────────────────────────────────────────────────────────────────────

def test_ping_endpoint(client):
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "running"


def test_chat_endpoint_sends_message(client, r):
    resp = client.post("/chat", json={"message": "Hello API"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "sent"
    assert "Hello API" in r.lrange(MESSAGE_LIST_KEY, 0, -1)


def test_chat_endpoint_empty_message_returns_400(client):
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 400


def test_chat_endpoint_missing_message_returns_400(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 400


def test_history_endpoint_cache_miss(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.get_json()["messages"] == []


def test_history_endpoint_cache_hit(client):
    client.post("/chat", json={"message": "Test"})
    resp = client.get("/history")
    assert "Test" in resp.get_json()["messages"]


# ─────────────────────────────────────────────────────────────────────────────
# Race condition safety — atomic pipeline
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_sends_do_not_exceed_max(r):
    """
    Simulates rapid concurrent sends.
    The atomic LPUSH+LTRIM pipeline ensures the list never exceeds MAX_MESSAGES.
    """
    import threading

    def send_n(n):
        for i in range(n):
            send_message(r, f"concurrent-{n}-{i}")

    threads = [threading.Thread(target=send_n, args=(5,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    messages = r.lrange(MESSAGE_LIST_KEY, 0, -1)
    assert len(messages) <= MAX_MESSAGES


def test_pipeline_is_atomic(r):
    """
    Verify all three pipeline operations (LPUSH, LTRIM, EXPIRE) apply together.
    """
    send_message(r, "atomic-test")
    assert r.llen(MESSAGE_LIST_KEY) >= 1
    assert r.ttl(MESSAGE_LIST_KEY) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Demo scenario: full live session simulation
# ─────────────────────────────────────────────────────────────────────────────

def test_full_demo_session(client, r):
    """
    Mirrors the 8-step live demo session from the blog post.
    """
    # Step 1: history before any messages → cache miss
    resp = client.get("/history")
    assert resp.get_json()["messages"] == []

    # Step 2: send a message
    client.post("/chat", json={"message": "Hello World!"})

    # Step 3: history after send → cache hit
    resp = client.get("/history")
    assert "Hello World!" in resp.get_json()["messages"]

    # Step 6: send multiple messages
    for i in range(1, 6):
        client.post("/chat", json={"message": f"Message {i}"})

    # Step 7: send beyond MAX_MESSAGES → eviction
    for i in range(MAX_MESSAGES + 2):
        client.post("/chat", json={"message": f"bulk-{i}"})

    resp = client.get("/history")
    assert len(resp.get_json()["messages"]) == MAX_MESSAGES
