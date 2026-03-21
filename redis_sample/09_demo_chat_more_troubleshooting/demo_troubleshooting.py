"""
demo_troubleshooting.py — Live demo script for:
"Demo Chat App with Redis: More Troubleshooting"

Walks through each troubleshooting scenario programmatically,
printing what you'd see in the console during a live session.

Requires a running Redis OR use fakeredis (default below).

    python demo_troubleshooting.py
"""

import time
import fakeredis   # swap for redis.Redis(host="localhost") for live demo

from chat_app import (
    send_message,
    get_history,
    MESSAGE_LIST_KEY,
    CACHE_TTL,
    MAX_MESSAGES,
)


def separator(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    r = fakeredis.FakeRedis(decode_responses=True)

    # ── Stage 1: Cache miss ──────────────────────────────────────────────────
    separator("Stage 1: Cache miss (GET /history before any messages)")
    msgs = get_history(r)
    print(f"  Result: {msgs}")
    print("  ✓ Expected: [CACHE MISS] No messages in cache")
    print("  Troubleshoot: key not yet populated — this is normal")

    # ── Stage 2: Send message ────────────────────────────────────────────────
    separator("Stage 2: POST /chat — send a message")
    send_message(r, "Hello troubleshooting demo!")
    print("  ✓ Expected: [PUB/SUB SENT] + [CACHE] Message cached")

    # ── Stage 3: Cache hit ───────────────────────────────────────────────────
    separator("Stage 3: GET /history — cache hit")
    msgs = get_history(r)
    print(f"  Result: {msgs}")
    print("  ✓ Expected: [CACHE HIT] Retrieved 1 messages")

    # ── Stage 4: TTL expiry ──────────────────────────────────────────────────
    separator(f"Stage 4: TTL expiry (CACHE_TTL={CACHE_TTL}s — simulated)")
    # Manually expire the key to simulate TTL
    r.delete(MESSAGE_LIST_KEY)
    msgs = get_history(r)
    print(f"  Result: {msgs}")
    print("  ✓ Expected: [CACHE MISS] No messages in cache")
    print(f"  Troubleshoot: run `redis-cli TTL {MESSAGE_LIST_KEY}` → -2 means expired")

    # ── Stage 5: Redis down ──────────────────────────────────────────────────
    separator("Stage 5: Redis down — connection error handling")
    import redis as redis_lib

    class DownRedis:
        def publish(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()
        def pipeline(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()
        def lrange(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()

    ok = send_message(DownRedis(), "test")
    history = get_history(DownRedis())
    print(f"  send_message returned: {ok}")
    print(f"  get_history returned: {history}")
    print("  ✓ Expected: [ERROR] Redis not available!")
    print("  Troubleshoot: run `redis-cli ping` — should return PONG")

    # ── Stage 6: Race conditions ─────────────────────────────────────────────
    separator("Stage 6: Multi-worker rapid sends")
    import threading
    r.flushall()

    def rapid_send(worker_id: int, count: int):
        for i in range(count):
            send_message(r, f"Worker{worker_id}-msg{i}")

    threads = [threading.Thread(target=rapid_send, args=(i, 3)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()

    msgs = get_history(r)
    print(f"  Total messages in cache: {len(msgs)} (max={MAX_MESSAGES})")
    print("  ✓ LPUSH+LTRIM pipeline is atomic — no partial updates")

    # ── Stage 7: Eviction ────────────────────────────────────────────────────
    separator(f"Stage 7: Eviction — send >{MAX_MESSAGES} messages")
    r.flushall()
    for i in range(MAX_MESSAGES + 5):
        send_message(r, f"bulk-msg-{i}")

    msgs = get_history(r)
    print(f"  Messages in cache: {len(msgs)} (LTRIM keeps last {MAX_MESSAGES})")
    print("  ✓ Oldest messages discarded — expected behaviour")

    print("\n=== Demo complete ===\n")


if __name__ == "__main__":
    main()
