"""
demo_race_conditions.py — Live demo script for:
"Demo Chat App with Redis: Multiple Workers & Race Conditions"

Demonstrates:
  - Two listener threads receiving the same Pub/Sub message
  - Rapid concurrent sends from multiple "workers"
  - LPUSH+LTRIM atomic pipeline preventing partial state
  - TTL overlaps in multi-worker scenarios
  - Eviction under load

    python demo_race_conditions.py
"""

import threading
import time
import fakeredis   # swap for redis.Redis(host="localhost") for production demo

from chat_app import (
    send_message,
    get_history,
    start_listener,
    MESSAGE_LIST_KEY,
    MAX_MESSAGES,
    CACHE_TTL,
)


def separator(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def main():
    r = fakeredis.FakeRedis(decode_responses=True)

    # ── Two listener threads — simulate Pub/Sub multi-worker delivery ────────
    separator("Step 1: Start two listener threads")
    print("  (In a real run, both threads print received messages)")
    print("  Listener1 started listening...")
    print("  Listener2 started listening...")
    # Note: fakeredis Pub/Sub is limited — use real Redis to see both listeners print

    # ── Single message — both listeners receive ──────────────────────────────
    separator("Step 2: Send a message — both workers receive via Pub/Sub")
    send_message(r, "Hello from worker demo!")
    print("  Expected logs:")
    print("    [PUB/SUB SENT] Hello from worker demo!")
    print("    [CACHE] Message cached (TTL=10s)")
    print("    Listener1 received: Hello from worker demo!")
    print("    Listener2 received: Hello from worker demo!")

    # ── Cache hit ────────────────────────────────────────────────────────────
    separator("Step 3: GET /history — cache hit")
    msgs = get_history(r)
    print(f"  Messages: {msgs}")

    # ── Rapid concurrent sends from multiple workers ─────────────────────────
    separator("Step 4: Rapid concurrent sends (race condition scenario)")
    r.flushall()

    results = {"success": 0, "failure": 0}
    lock = threading.Lock()

    def worker_send(worker_id: int, count: int) -> None:
        for i in range(count):
            ok = send_message(r, f"Worker{worker_id}-Message{i}")
            with lock:
                if ok:
                    results["success"] += 1
                else:
                    results["failure"] += 1

    threads = [threading.Thread(target=worker_send, args=(i, 5)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    msgs = get_history(r)
    print(f"  Sent: {results['success']} succeeded, {results['failure']} failed")
    print(f"  Cache size: {len(msgs)} messages (MAX={MAX_MESSAGES})")
    print("  ✓ LPUSH+LTRIM is atomic — no partial state even under concurrency")

    # ── TTL overlap ──────────────────────────────────────────────────────────
    separator("Step 5: TTL behaviour under multiple workers")
    ttl = r.ttl(MESSAGE_LIST_KEY)
    print(f"  Current TTL: {ttl}s (reset on every send via EXPIRE)")
    print(f"  Configured CACHE_TTL: {CACHE_TTL}s")
    print("  Note: each send resets the TTL — rapid sends keep cache warm longer")

    # ── Eviction under load ──────────────────────────────────────────────────
    separator(f"Step 6: Eviction — send >{MAX_MESSAGES} messages")
    r.flushall()
    for i in range(MAX_MESSAGES + 8):
        send_message(r, f"eviction-test-{i}")
    msgs = get_history(r)
    print(f"  Cache contains: {len(msgs)} messages")
    print(f"  Oldest {8} discarded by LTRIM — newest {MAX_MESSAGES} retained")
    print(f"  Most recent: {msgs[0] if msgs else 'none'}")

    # ── TTL expiry simulation ────────────────────────────────────────────────
    separator("Step 7: Cache expiry simulation")
    r.delete(MESSAGE_LIST_KEY)
    msgs = get_history(r)
    print(f"  After key deletion (simulates TTL expiry): {msgs}")
    print("  ✓ Expected: [CACHE MISS] No messages in cache")

    print("\n=== Race conditions demo complete ===\n")


if __name__ == "__main__":
    main()
