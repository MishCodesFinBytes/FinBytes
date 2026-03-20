"""
redis_gunicorn_cache.py — Runnable demo for:
"Redis, Gunicorn, and Cache Issues"

Demonstrates:
  - Read-through caching with TTL (the enterprise baseline)
  - Stale key scanning and cleanup
  - Cache stampede protection via distributed lock
  - Worker-local vs shared cache mental model

Run with a local Redis:
    redis-server
    python redis_gunicorn_cache.py

Tests (no Redis required):
    pytest test_redis_gunicorn_cache.py -v
"""

import json
import logging
import time
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Simulated database
# ─────────────────────────────────────────────────────────────────────────────

_db = {
    1: {"user_id": 1, "name": "Alice", "balance": 1500.00},
    2: {"user_id": 2, "name": "Bob",   "balance": 350.00},
}

def db_fetch_user(user_id: int) -> dict | None:
    time.sleep(0.02)
    return _db.get(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: Basic read-through cache (enterprise baseline)
# ─────────────────────────────────────────────────────────────────────────────

def get_profile(user_id: int, r: redis.Redis, ttl: int = 300) -> dict | None:
    """
    Read-through caching:
      1. Hit Redis
      2. On miss: fetch from DB, write back with TTL
    All Gunicorn workers share the same Redis, so all see the same value.
    """
    key = f"user:{user_id}"
    cached = r.get(key)

    if cached is not None:
        logging.info(f"[CACHE HIT]  {key}")
        return json.loads(cached)

    logging.info(f"[CACHE MISS] {key} — querying DB")
    profile = db_fetch_user(user_id)

    if profile:
        r.set(key, json.dumps(profile), ex=ttl)
        logging.info(f"[CACHE SET]  {key} TTL={ttl}s")

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: Cache stampede protection (distributed lock)
# ─────────────────────────────────────────────────────────────────────────────

def get_profile_safe(user_id: int, r: redis.Redis, ttl: int = 300) -> dict | None:
    """
    Stampede-safe read-through cache.
    When a key expires, only ONE worker refreshes it; others serve the
    (briefly stale) value or wait.

    nx=True on the lock key means only the first caller sets it.
    """
    key = f"user:{user_id}"
    lock_key = f"{key}:lock"

    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)

    # Try to acquire the refresh lock (5s expiry prevents dead locks)
    if r.set(lock_key, "1", nx=True, ex=5):
        try:
            profile = db_fetch_user(user_id)
            if profile:
                r.set(key, json.dumps(profile), ex=ttl)
                logging.info(f"[LOCK REFRESH] {key}")
            return profile
        finally:
            r.delete(lock_key)

    # Another worker is refreshing — serve whatever is there (may be None briefly)
    logging.info(f"[LOCK WAIT]  {key} — another worker is refreshing")
    cached = r.get(key)
    return json.loads(cached) if cached else None


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: Stale key inspection and cleanup
# ─────────────────────────────────────────────────────────────────────────────

def scan_stale_keys(r: redis.Redis, pattern: str = "user:*") -> list[dict]:
    """
    Non-destructive scan: lists keys with their TTL status.
    Safe to run from app code without sysadmin rights.
    """
    results = []
    for key in r.scan_iter(pattern):
        ttl = r.ttl(key)
        status = "ok" if ttl > 0 else ("no-expiry" if ttl == -1 else "missing")
        results.append({"key": key, "ttl": ttl, "status": status})
        logging.info(f"[SCAN] {key} TTL={ttl} ({status})")
    return results


def delete_stale_keys(r: redis.Redis, pattern: str = "user:*") -> int:
    """Delete all keys matching a pattern. Use carefully in production."""
    deleted = 0
    for key in r.scan_iter(pattern):
        r.delete(key)
        deleted += 1
        logging.info(f"[DELETE] {key}")
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: Lazy rebuild (simulate what happens after a flush)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_flush_and_rebuild(r: redis.Redis) -> None:
    """
    After a FLUSHALL or Redis restart:
    - First requests are cache misses (expected)
    - DB handles the load temporarily
    - Cache warms up automatically on access
    """
    logging.info("[FLUSH] Simulating Redis flush (deleting all user keys)")
    delete_stale_keys(r, "user:*")

    logging.info("[REBUILD] First accesses after flush — expect cache misses")
    for uid in [1, 2]:
        get_profile(uid, r)

    logging.info("[REBUILD] Second accesses — expect cache hits")
    for uid in [1, 2]:
        get_profile(uid, r)


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    r = redis.Redis(host="localhost", port=6379, db=1, decode_responses=True)

    print("\n=== Redis + Gunicorn Cache Demo ===\n")

    print("-- Basic read-through --")
    get_profile(1, r)
    get_profile(1, r)

    print("\n-- Stale key scan --")
    scan_stale_keys(r)

    print("\n-- Flush + rebuild simulation --")
    simulate_flush_and_rebuild(r)

    # Cleanup
    r.flushdb()
    print("\nDone. Test DB flushed.")


if __name__ == "__main__":
    main()
