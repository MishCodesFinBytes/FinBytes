"""
redis_intro_checks.py — Runnable demo for:
"Redis – intro and checks"

Covers:
  - Basic connectivity and health checks (no sysadmin rights needed)
  - Good vs bad caching patterns (lru_cache vs Redis)
  - TTL inspection
  - Read-through with DB fallback
  - Simulating a Redis flush and rebuild

Run with a local Redis:
    redis-server
    python redis_intro_checks.py

Tests (no Redis required):
    pytest test_redis_intro_checks.py -v
"""

import json
import logging
import time
from functools import lru_cache
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Simulated database and helpers
# ─────────────────────────────────────────────────────────────────────────────

_db = {
    1: {"user_id": 1, "name": "Alice", "balance": 1500.00},
    2: {"user_id": 2, "name": "Bob",   "balance": 350.00},
}

def db_fetch_user(user_id: int) -> dict | None:
    time.sleep(0.02)
    return _db.get(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Health checks — safe to run without sysadmin rights
# ─────────────────────────────────────────────────────────────────────────────

def check_connectivity(r: redis.Redis) -> bool:
    """Is Redis reachable?"""
    try:
        result = r.ping()
        logging.info(f"[PING] Redis alive: {result}")
        return result
    except redis.exceptions.ConnectionError as e:
        logging.error(f"[PING] Redis unreachable: {e}")
        return False


def check_key(r: redis.Redis, key: str) -> None:
    """Inspect a key's value and TTL."""
    value = r.get(key)
    ttl = r.ttl(key)
    ttl_label = f"{ttl}s" if ttl > 0 else ("no-expiry" if ttl == -1 else "missing")
    logging.info(f"[KEY]  {key}: value={value!r}  TTL={ttl_label}")


def check_hit_rate(r: redis.Redis) -> float:
    """Calculate cache hit rate from Redis stats."""
    info = r.info("stats")
    hits = info.get("keyspace_hits", 0)
    misses = info.get("keyspace_misses", 0)
    total = hits + misses
    rate = round(hits / total * 100, 1) if total > 0 else 0.0
    logging.info(f"[STATS] hits={hits} misses={misses} hit_rate={rate}%")
    return rate


def check_memory(r: redis.Redis) -> dict:
    """Report memory usage."""
    info = r.info("memory")
    used = info.get("used_memory_human", "?")
    peak = info.get("used_memory_peak_human", "?")
    logging.info(f"[MEM]  used={used}  peak={peak}")
    return {"used": used, "peak": peak}


# ─────────────────────────────────────────────────────────────────────────────
# GOOD pattern: lru_cache for pure functions, Redis for shared data
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def format_currency(amount: float) -> str:
    """Pure function — safe for lru_cache. Same input → same output always."""
    return f"£{amount:,.2f}"


def get_user_balance(user_id: int, r: redis.Redis) -> str:
    """
    GOOD: Redis holds the mutable balance; lru_cache only formats the value.
    All workers see the same balance via Redis.
    """
    raw = r.get(f"balance:{user_id}")
    balance = float(raw) if raw else 0.0
    return format_currency(balance)


# ─────────────────────────────────────────────────────────────────────────────
# BAD pattern — shown for comparison, NOT used below
# ─────────────────────────────────────────────────────────────────────────────

# @lru_cache(maxsize=1024)
# def get_user_balance_bad(user_id, r):
#     return r.get(f"balance:{user_id}")
#
# This breaks across workers:
# - Server A caches old balance locally
# - Redis gets updated
# - Server A still serves stale value indefinitely


# ─────────────────────────────────────────────────────────────────────────────
# Read-through cache with rebuild
# ─────────────────────────────────────────────────────────────────────────────

def get_user_profile(user_id: int, r: redis.Redis, ttl: int = 300) -> dict | None:
    key = f"user:{user_id}"
    cached = r.get(key)

    if cached:
        logging.info(f"[CACHE HIT]  {key}")
        return json.loads(cached)

    logging.info(f"[CACHE MISS] {key}")
    profile = db_fetch_user(user_id)

    if profile:
        r.set(key, json.dumps(profile), ex=ttl)

    return profile


def simulate_flush_and_rebuild(r: redis.Redis) -> None:
    """
    After Redis flushes, the system degrades gracefully:
    - First requests miss cache (expected)
    - DB handles the load temporarily
    - Cache warms up automatically
    """
    logging.info("[FLUSH] Clearing all user keys")
    for key in r.scan_iter("user:*"):
        r.delete(key)

    logging.info("[REBUILD] Accessing users — cache misses expected")
    get_user_profile(1, r)
    get_user_profile(2, r)

    logging.info("[REBUILD] Second access — cache hits expected")
    get_user_profile(1, r)
    get_user_profile(2, r)


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    r = redis.Redis(host="localhost", port=6379, db=3, decode_responses=True)

    print("\n=== Redis Intro & Checks Demo ===\n")

    print("-- Health checks --")
    if not check_connectivity(r):
        print("Redis not running. Start with: redis-server")
        return

    print("\n-- Warm up cache --")
    get_user_profile(1, r)
    get_user_profile(2, r)

    print("\n-- Key inspection --")
    check_key(r, "user:1")
    check_key(r, "user:missing")

    print("\n-- Stats --")
    get_user_profile(1, r)
    get_user_profile(1, r)
    check_hit_rate(r)

    print("\n-- Memory --")
    check_memory(r)

    print("\n-- Pure function (lru_cache) --")
    r.set("balance:1", "1500.00")
    print(f"   Balance: {get_user_balance(1, r)}")

    print("\n-- Flush & rebuild --")
    simulate_flush_and_rebuild(r)

    r.flushdb()
    print("\nDone.")


if __name__ == "__main__":
    main()
