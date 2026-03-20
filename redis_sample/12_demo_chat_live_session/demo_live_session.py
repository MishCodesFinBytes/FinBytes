"""
demo_live_session.py — Live demo script for:
"Demo Chat App with Redis: Live Session Simulation"

Runs the full 10-step live session from the blog's cheat sheet,
printing expected logs and talking points at each step.

    python demo_live_session.py
"""

import threading
import time
import fakeredis   # swap for redis.Redis(host="localhost") for live demo

from chat_app import (
    send_message,
    get_history,
    MESSAGE_LIST_KEY,
    MAX_MESSAGES,
    CACHE_TTL,
)

import redis as redis_lib


class DownRedis:
    def publish(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()
    def pipeline(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()
    def lrange(self, *a, **kw): raise redis_lib.exceptions.ConnectionError()


def step(number: int, title: str, endpoint: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  Step {number}: {title}")
    print(f"  Endpoint: {endpoint}")
    print(f"{'═'*60}")


def expected(log: str) -> None:
    print(f"  Expected log: {log}")


def talking_point(text: str) -> None:
    print(f"  💬 {text}")


def main():
    r = fakeredis.FakeRedis(decode_responses=True)
    print("\n🎤  Redis Chat Demo — Live Session Simulation")
    print("    Mirrors the 10-step cheat sheet from the blog post\n")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    step(1, "Cache miss before any messages", "GET /history")
    msgs = get_history(r)
    expected("[CACHE MISS] No messages in cache")
    talking_point("Cache miss — key not yet populated. Normal first-access behaviour.")
    print(f"  Result: {msgs}")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    step(2, "Send first message", "POST /chat {\"message\": \"Hello\"}")
    ok = send_message(r, "Hello")
    expected("[PUB/SUB SENT] Hello  |  [CACHE] Message cached (TTL=10s)  |  Listener1/2 received: Hello")
    talking_point("Pub/Sub delivers to all subscribers. List cache stores for history.")
    print(f"  send_message returned: {ok}")

    # ── Step 3 ───────────────────────────────────────────────────────────────
    step(3, "Cache hit after message sent", "GET /history")
    msgs = get_history(r)
    expected(f"[CACHE HIT] Retrieved {len(msgs)} messages")
    talking_point("Redis returns message immediately — no DB involved.")
    print(f"  Result: {msgs}")

    # ── Step 4 ───────────────────────────────────────────────────────────────
    step(4, f"TTL expiry (simulated — real TTL={CACHE_TTL}s)", "GET /history after wait")
    r.delete(MESSAGE_LIST_KEY)
    msgs = get_history(r)
    expected("[CACHE MISS] No messages in cache")
    talking_point(f"After {CACHE_TTL}s the key expires. Important to set TTL appropriately for your use case.")
    print(f"  Result: {msgs}")

    # ── Step 5 ───────────────────────────────────────────────────────────────
    step(5, "Redis down — error handling", "POST /chat after stopping Redis")
    ok = send_message(DownRedis(), "test message")
    expected("[ERROR] Redis not available! Message not sent.")
    talking_point("App handles Redis outage gracefully — error logged, no crash.")
    print(f"  send_message returned: {ok} (False = graceful failure)")

    # ── Step 6 ───────────────────────────────────────────────────────────────
    step(6, "Multi-worker rapid sends", "Multiple POST /chat requests")
    r.flushall()
    results = {"n": 0}

    def worker_send(wid):
        for i in range(3):
            if send_message(r, f"Worker{wid}-msg{i}"):
                results["n"] += 1

    threads = [threading.Thread(target=worker_send, args=(i,)) for i in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()

    msgs = get_history(r)
    expected("[PUB/SUB SENT] Message N  |  [Listener1/2 received: Message N]  |  [CACHE] Message cached")
    talking_point("Atomic LPUSH+LTRIM pipeline prevents partial state across workers.")
    print(f"  {results['n']} messages sent. Cache holds {len(msgs)} (MAX={MAX_MESSAGES})")

    # ── Step 7 ───────────────────────────────────────────────────────────────
    step(7, f"Eviction — send >{MAX_MESSAGES} messages", "POST /chat × 15")
    r.flushall()
    for i in range(MAX_MESSAGES + 5):
        send_message(r, f"bulk-{i}")
    msgs = get_history(r)
    expected(f"[CACHE HIT] Retrieved {MAX_MESSAGES} messages")
    talking_point(f"LTRIM keeps only the last {MAX_MESSAGES} messages. Older ones are discarded automatically.")
    print(f"  Cache size: {len(msgs)}")

    # ── Step 8 ───────────────────────────────────────────────────────────────
    step(8, "Redis outage mid-demo", "Stop Redis → POST /chat")
    ok = send_message(DownRedis(), "during outage")
    expected("[ERROR] Redis not available!")
    talking_point("System logs the error and moves on — no crash, no data corruption.")
    print(f"  send_message returned: {ok}")

    # ── Step 9 ───────────────────────────────────────────────────────────────
    step(9, "New listener starts after messages sent", "GET /history from new subscriber")
    r.flushall()
    send_message(r, "pre-existing message")
    # A new listener joining now would miss the Pub/Sub event above,
    # but can still retrieve history from the list cache
    msgs = get_history(r)
    expected("[CACHE HIT] Retrieved 1 messages  (Pub/Sub missed but list available)")
    talking_point("Pub/Sub is ephemeral — late subscribers miss past events. History list compensates.")
    print(f"  History available: {msgs}")

    # ── Step 10 ──────────────────────────────────────────────────────────────
    step(10, "Rapid sends + TTL — mixed hits and misses", "Interleaved sends and reads")
    r.flushall()
    for i in range(3):
        send_message(r, f"rapid-{i}")
        get_history(r)   # interleave reads
    r.delete(MESSAGE_LIST_KEY)
    get_history(r)       # forced miss

    expected("[CACHE HIT] / [CACHE MISS] interleaved depending on timing")
    talking_point("In multi-worker environments, overlapping TTLs can cause mixed hit/miss patterns.")

    print(f"\n{'═'*60}")
    print("  🏁  Live session simulation complete")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
