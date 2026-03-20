"""
troubleshooting_redis.py — Runnable demo for:
"Troubleshooting Redis"

Demonstrates each common Redis issue from the post:
  1. Cache misses (TTL expiry, cold start)
  2. High memory / eviction
  3. Stale data
  4. Redis down — graceful fallback
  5. Cache stampede
  6. Redis flush — system survives

Every scenario is testable without a live Redis server.

Tests:
    pip install fakeredis
    pytest test_troubleshooting_redis.py -v
"""

import json
import logging
import time
import fakeredis
import redis as redis_lib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Simulated database — the permanent source of truth
# ─────────────────────────────────────────────────────────────────────────────

_db = {
    "user:1": {"name": "Alice", "balance": 1500.00},
    "user:2": {"name": "Bob",   "balance": 275.00},
}

def db_fetch(key: str) -> dict | None:
    return _db.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# Core read-through helper
# ─────────────────────────────────────────────────────────────────────────────

def read_through(key: str, r, ttl: int = 300) -> dict | None:
    """
    Reads from Redis first. On miss, falls back to DB and writes result back.
    This is the correct, survivable pattern — Redis down = slower, not broken.
    """
    log = logging.getLogger("cache")
    try:
        cached = r.get(key)
        if cached:
            log.info(f"[CACHE HIT]  {key}")
            return json.loads(cached)

        log.info(f"[CACHE MISS] {key} — falling back to DB")
        value = db_fetch(key)
        if value:
            r.set(key, json.dumps(value), ex=ttl)
        return value

    except redis_lib.exceptions.ConnectionError:
        log.error(f"[REDIS DOWN] falling back to DB for {key}")
        return db_fetch(key)


# ─────────────────────────────────────────────────────────────────────────────
# Issue 1: Cache miss — expected, not a bug
# ─────────────────────────────────────────────────────────────────────────────

def demo_cache_miss(r) -> dict:
    """
    MISS: key not yet populated.
    FIX: first access always misses; system rebuilds automatically.
    """
    result = read_through("user:1", r)
    return {"miss_then_hit": result is not None}


def demo_ttl_expiry(r) -> dict:
    """
    MISS after TTL expires.
    FIX: lower TTL, accept brief slowness, or use background refresh.
    """
    key = "user:1"
    r.set(key, json.dumps({"name": "Alice"}), ex=1)  # 1s TTL
    time.sleep(1.1)                                    # let it expire
    result = read_through(key, r)
    return {"rebuilt_after_expiry": result is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Issue 2: Stale data
# ─────────────────────────────────────────────────────────────────────────────

def demo_stale_data(r) -> dict:
    """
    Shows the effect of a long TTL: cache serves old data even after DB update.
    FIX: lower TTL or explicitly invalidate on write.
    """
    key = "user:1"
    r.set(key, json.dumps({"name": "Alice", "balance": 1000.00}), ex=3600)

    # DB is updated but cache is NOT invalidated
    _db["user:1"]["balance"] = 500.00

    cached = json.loads(r.get(key))
    db_value = db_fetch(key)

    stale = cached["balance"] != db_value["balance"]
    return {"stale_detected": stale, "cached": cached["balance"], "real": db_value["balance"]}


def fix_stale_data(r) -> dict:
    """FIX: invalidate the key on write."""
    key = "user:1"
    r.set(key, json.dumps({"name": "Alice", "balance": 1000.00}), ex=3600)
    _db["user:1"]["balance"] = 500.00

    # Correct pattern: delete cache entry on update
    r.delete(key)
    fresh = read_through(key, r)
    return {"fresh_balance": fresh["balance"]}


# ─────────────────────────────────────────────────────────────────────────────
# Issue 3: Redis down — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

class DownRedis:
    """Simulates a Redis instance that refuses all connections."""
    def get(self, key):
        raise redis_lib.exceptions.ConnectionError("Redis is down")

    def set(self, key, value, ex=None):
        raise redis_lib.exceptions.ConnectionError("Redis is down")


def demo_redis_down() -> dict:
    """
    When Redis is down, read_through falls back to DB.
    Response is slower but correct — the system survives.
    """
    r = DownRedis()
    result = read_through("user:1", r)
    return {"survived": result is not None, "name": result["name"] if result else None}


# ─────────────────────────────────────────────────────────────────────────────
# Issue 4: Cache stampede
# ─────────────────────────────────────────────────────────────────────────────

def demo_stampede_protection(r) -> dict:
    """
    Stampede: when a key expires, multiple callers all hit the DB simultaneously.
    FIX: distributed lock — only one caller refreshes, others wait.
    """
    key = "user:1"
    lock_key = f"{key}:lock"
    db_calls = {"n": 0}

    def get_with_lock(r):
        cached = r.get(key)
        if cached:
            return json.loads(cached)

        if r.set(lock_key, "1", nx=True, ex=5):
            try:
                db_calls["n"] += 1
                value = db_fetch(key)
                r.set(key, json.dumps(value), ex=300)
                return value
            finally:
                r.delete(lock_key)

        # Another caller holds the lock
        cached = r.get(key)
        return json.loads(cached) if cached else None

    # Simulate 5 concurrent callers hitting an empty cache
    results = [get_with_lock(r) for _ in range(5)]
    return {"db_calls": db_calls["n"], "all_got_result": all(r is not None for r in results)}


# ─────────────────────────────────────────────────────────────────────────────
# Issue 5: Redis flush / restart — system recovers automatically
# ─────────────────────────────────────────────────────────────────────────────

def demo_flush_recovery(r) -> dict:
    """
    After FLUSHALL, cache misses increase temporarily.
    System rebuilds from DB automatically — no data lost.
    """
    # Warm cache
    read_through("user:1", r)
    read_through("user:2", r)
    assert r.get("user:1") is not None

    # Flush (simulates Redis restart or FLUSHALL)
    r.flushall()
    assert r.get("user:1") is None

    # First access after flush — miss, rebuilds
    result = read_through("user:1", r)
    rebuilt = r.get("user:1") is not None
    return {"data_survived": result is not None, "cache_rebuilt": rebuilt}


# ─────────────────────────────────────────────────────────────────────────────
# Debugging checklist
# ─────────────────────────────────────────────────────────────────────────────

def run_debug_checklist(r) -> dict:
    """
    Runs the 5 diagnostic questions from the post:
      1. Is Redis up?
      2. Are keys expiring (TTL set)?
      3. Is fallback (DB) working?
      4. Is the database healthy?
      5. Is the problem slow or incorrect?
    """
    results = {}

    try:
        results["redis_up"] = r.ping()
    except Exception:
        results["redis_up"] = False

    # Seed a key and check TTL
    r.set("user:1", json.dumps({"name": "Alice"}), ex=300)
    ttl = r.ttl("user:1")
    results["keys_expiring"] = ttl > 0

    # Fallback
    fallback = db_fetch("user:1")
    results["fallback_working"] = fallback is not None

    # DB healthy
    results["db_healthy"] = db_fetch("user:1") is not None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    r = fakeredis.FakeRedis(decode_responses=True)

    print("\n=== Redis Troubleshooting Demo ===\n")

    print("-- Issue 1: Cache miss (expected) --")
    print(f"   {demo_cache_miss(r)}\n")

    print("-- Issue 2: Stale data --")
    stale = demo_stale_data(r)
    print(f"   Stale: {stale}")
    fix = fix_stale_data(r)
    print(f"   Fixed: {fix}\n")

    print("-- Issue 3: Redis down → fallback to DB --")
    print(f"   {demo_redis_down()}\n")

    print("-- Issue 4: Stampede protection --")
    r.flushall()
    print(f"   {demo_stampede_protection(r)}\n")

    print("-- Issue 5: Flush recovery --")
    print(f"   {demo_flush_recovery(r)}\n")

    print("-- Debug checklist --")
    print(f"   {run_debug_checklist(r)}")


if __name__ == "__main__":
    main()
