"""
redis_basics.py — Runnable demo for:
"Don't Be Scared of Redis: What It Really Means When Your Company Uses It"

Demonstrates the core Redis pattern every engineer needs to know:
  - read-through caching (check Redis first, fall back to DB)
  - TTL-based expiry
  - safe lru_cache for pure functions only

Run with a local Redis server:
    redis-server
    python redis_basics.py

Or run the tests (no Redis required):
    pytest test_redis_basics.py -v
"""

import time
import redis
from functools import lru_cache


# ─────────────────────────────────────────────────────────────────────────────
# Simulated database (replaces a real DB for demo purposes)
# ─────────────────────────────────────────────────────────────────────────────

_fake_db = {
    1: {"name": "Alice", "balance": 1500.00},
    2: {"name": "Bob",   "balance": 275.50},
    3: {"name": "Carol", "balance": 9820.00},
}

def db_fetch_user(user_id: int) -> dict | None:
    """Simulate a slow database query."""
    time.sleep(0.05)  # artificial latency
    return _fake_db.get(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# GOOD: lru_cache for pure deterministic functions only
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def format_currency(amount: float) -> str:
    """
    Pure function — same input always gives same output.
    Safe to cache locally. No Redis needed.
    """
    return f"£{amount:,.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# Core pattern: read-through cache with Redis
# ─────────────────────────────────────────────────────────────────────────────

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def get_user_profile(user_id: int, r: redis.Redis) -> dict | None:
    """
    The most common Redis pattern in enterprise systems.

    1. Check Redis first (fast, shared)
    2. On miss: fetch from DB, write back to Redis with TTL
    3. Next caller hits Redis (cache warm)
    """
    key = f"user:{user_id}"

    cached = r.get(key)
    if cached is not None:
        logging.info(f"[CACHE HIT]  key={key}")
        return json.loads(cached)

    logging.info(f"[CACHE MISS] key={key} — fetching from DB")
    profile = db_fetch_user(user_id)

    if profile is not None:
        r.set(key, json.dumps(profile), ex=300)  # cache for 5 minutes
        logging.info(f"[CACHE SET]  key={key} TTL=300s")

    return profile


def invalidate_user(user_id: int, r: redis.Redis) -> None:
    """Force a cache miss on next access — e.g. after a balance update."""
    key = f"user:{user_id}"
    r.delete(key)
    logging.info(f"[CACHE DEL]  key={key}")


def check_redis_health(r: redis.Redis) -> bool:
    """Quick connectivity check — safe to call from app startup."""
    try:
        return r.ping()
    except redis.exceptions.ConnectionError:
        logging.error("[REDIS DOWN] Cannot reach Redis")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Runnable main demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    print("\n=== Redis Basics Demo ===\n")

    if not check_redis_health(r):
        print("Redis is not running. Start it with: redis-server")
        return

    # First access — cache miss, DB fetch
    print("-- First access (expect CACHE MISS) --")
    profile = get_user_profile(1, r)
    print(f"   Result: {profile}\n")

    # Second access — cache hit
    print("-- Second access (expect CACHE HIT) --")
    profile = get_user_profile(1, r)
    print(f"   Result: {profile}\n")

    # Pure function cached locally
    print("-- Pure function (lru_cache, no Redis) --")
    print(f"   {format_currency(1500.0)}")
    print(f"   {format_currency(275.5)}\n")

    # TTL inspection
    key = "user:1"
    ttl = r.ttl(key)
    print(f"-- TTL check: key={key} has {ttl}s remaining --\n")

    # Invalidation
    print("-- Invalidate user 1 (simulating balance update) --")
    invalidate_user(1, r)

    print("-- Access after invalidation (expect CACHE MISS) --")
    get_user_profile(1, r)


if __name__ == "__main__":
    main()
