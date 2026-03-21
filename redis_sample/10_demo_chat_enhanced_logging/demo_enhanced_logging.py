"""
demo_enhanced_logging.py — Live demo script for:
"Demo Chat App with Redis: Enhanced Logging Demo"

Demonstrates what every log line means and when it appears.
All events are visible in the console — cache hits/misses,
Pub/Sub delivery, TTL set, errors.

    python demo_enhanced_logging.py
"""

import logging
import fakeredis   # swap for redis.Redis(host="localhost") for live demo

from chat_app import send_message, get_history, MESSAGE_LIST_KEY

# Set logging level to DEBUG to see all events
logging.getLogger().setLevel(logging.DEBUG)


def annotate(label: str, fn, *args, **kwargs):
    """Run a function and print what log output to expect."""
    print(f"\n  → {label}")
    result = fn(*args, **kwargs)
    return result


def main():
    r = fakeredis.FakeRedis(decode_responses=True)
    print("\n=== Enhanced Logging Demo ===")
    print("Watch the [HH:MM:SS] prefixed log lines below each action.\n")

    # ── Cache miss ───────────────────────────────────────────────────────────
    print("┌─ Action: GET /history (empty cache)")
    annotate("Expected: [CACHE MISS] No messages in cache", get_history, r)
    print("└─ Explanation: No key exists yet — first access always misses\n")

    # ── Send message ─────────────────────────────────────────────────────────
    print("┌─ Action: POST /chat {\"message\": \"Hello World\"}")
    annotate(
        "Expected: [PUB/SUB SENT] Hello World  |  [CACHE] Message cached (TTL=10s)",
        send_message, r, "Hello World",
    )
    print("│  PUB/SUB SENT  → broadcast to all current subscribers")
    print("│  CACHE         → LPUSH + LTRIM + EXPIRE applied atomically")
    print("└─ Note: [Listener1/2 received: Hello World] appears in a real server\n")

    # ── Cache hit ────────────────────────────────────────────────────────────
    print("┌─ Action: GET /history (cache warm)")
    msgs = annotate("Expected: [CACHE HIT] Retrieved 1 messages", get_history, r)
    print(f"│  Messages returned: {msgs}")
    print("└─ Explanation: Redis list returned directly — no DB hit\n")

    # ── TTL expiry ───────────────────────────────────────────────────────────
    print("┌─ Action: Simulate TTL expiry (deleting key)")
    r.delete(MESSAGE_LIST_KEY)
    annotate("Expected: [CACHE MISS] No messages in cache", get_history, r)
    print("└─ In a real demo: wait 10s then GET /history\n")

    # ── Redis down ───────────────────────────────────────────────────────────
    print("┌─ Action: Redis down simulation")
    import redis as redis_lib

    class DownRedis:
        def publish(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()
        def pipeline(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()
        def lrange(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()

    annotate(
        "Expected: [ERROR] Redis not available! Message not sent.",
        send_message, DownRedis(), "will fail",
    )
    print("└─ Graceful: error logged, False returned, app continues\n")

    # ── Multiple messages → eviction ─────────────────────────────────────────
    print("┌─ Action: Send 15 messages (MAX_MESSAGES=10)")
    for i in range(15):
        send_message(r, f"msg-{i}")
    msgs = get_history(r)
    print(f"│  Messages in cache: {len(msgs)} (oldest 5 discarded by LTRIM)")
    print("└─ Expected: [CACHE HIT] Retrieved 10 messages\n")

    print("=== Logging demo complete ===\n")


if __name__ == "__main__":
    main()
