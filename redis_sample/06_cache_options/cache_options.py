"""
cache_options.py — Runnable demo for:
"Cache – Options and varieties available"

Five caching patterns demonstrated side by side:
  1. Redis with TTL        — enterprise baseline
  2. Redis + stampede lock — prevents DB meltdown on expiry
  3. TTLCache (single-process only, honest alternative to lru_cache)
  4. Async background refresh — serve stale, refresh behind the scenes
  5. Anti-pattern: lru_cache wrapping Redis — shown and explained

Tests (no live Redis needed):
    pip install fakeredis cachetools
    pytest test_cache_options.py -v
"""

import json
import logging
import time
import threading
from functools import lru_cache
import fakeredis
from cachetools import TTLCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Simulated database
# ─────────────────────────────────────────────────────────────────────────────

_db = {
    "user:1": {"name": "Alice", "balance": 1500.00},
    "user:2": {"name": "Bob",   "balance": 350.00},
}
_db_call_count = {"n": 0}

def db_fetch(key: str) -> dict | None:
    _db_call_count["n"] += 1
    time.sleep(0.01)
    return _db.get(key)

def db_call_count() -> int:
    return _db_call_count["n"]

def reset_db_counter():
    _db_call_count["n"] = 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Redis with TTL — the boring but correct enterprise default
# ─────────────────────────────────────────────────────────────────────────────

def get_with_ttl(key: str, r: fakeredis.FakeRedis, ttl: int = 300) -> dict | None:
    """
    Read-through cache. All servers share Redis.
    Safe, predictable, and easy to reason about.
    """
    cached = r.get(key)
    if cached:
        logging.getLogger("ttl").info(f"[HIT]  {key}")
        return json.loads(cached)

    logging.getLogger("ttl").info(f"[MISS] {key} → DB")
    value = db_fetch(key)
    if value:
        r.set(key, json.dumps(value), ex=ttl)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# 2. Redis + stampede lock (dogpile protection)
# ─────────────────────────────────────────────────────────────────────────────

def get_with_lock(key: str, r: fakeredis.FakeRedis, ttl: int = 300) -> dict | None:
    """
    When a cache entry expires, only one caller refreshes it.
    Others get the old value (or None briefly) rather than all hitting the DB.
    """
    log = logging.getLogger("lock")
    lock_key = f"{key}:lock"

    cached = r.get(key)
    if cached:
        return json.loads(cached)

    # nx=True: only the first caller sets the lock
    if r.set(lock_key, "1", nx=True, ex=5):
        try:
            value = db_fetch(key)
            if value:
                r.set(key, json.dumps(value), ex=ttl)
            log.info(f"[LOCK REFRESH] {key}")
            return value
        finally:
            r.delete(lock_key)

    # Another worker holds the lock — serve whatever is available
    log.info(f"[LOCK WAIT]  {key}")
    cached = r.get(key)
    return json.loads(cached) if cached else None


# ─────────────────────────────────────────────────────────────────────────────
# 3. TTLCache — single-process only, but at least it expires
# ─────────────────────────────────────────────────────────────────────────────

_ttl_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

def get_with_ttlcache(key: str) -> dict | None:
    """
    In-process TTLCache (cachetools). Acceptable for:
      - single-server deployments
      - background jobs
      - batch pipelines

    NOT safe for multi-server/multi-worker where correctness matters.
    At least it expires — unlike lru_cache.
    """
    log = logging.getLogger("ttlcache")
    if key in _ttl_cache:
        log.info(f"[HIT]  {key}")
        return _ttl_cache[key]

    value = db_fetch(key)
    if value:
        _ttl_cache[key] = value
    log.info(f"[MISS] {key}")
    return value


# ─────────────────────────────────────────────────────────────────────────────
# 4. Async background refresh — serve stale, update behind the scenes
# ─────────────────────────────────────────────────────────────────────────────

_refresh_queue: set[str] = set()

def _background_refresh(key: str, r: fakeredis.FakeRedis, ttl: int) -> None:
    """Worker that refreshes a key in the background."""
    value = db_fetch(key)
    if value:
        r.set(key, json.dumps(value), ex=ttl)
    _refresh_queue.discard(key)
    logging.getLogger("async").info(f"[REFRESHED] {key}")


def get_with_background_refresh(
    key: str, r: fakeredis.FakeRedis, ttl: int = 300, refresh_threshold: int = 30
) -> dict | None:
    """
    Serve cached value immediately (even if slightly stale).
    Trigger a background refresh when TTL drops below threshold.
    Users stay fast; data converges quickly.
    """
    log = logging.getLogger("async")
    cached = r.get(key)

    if cached is None:
        # Cold start — must wait for DB
        log.info(f"[COLD]  {key}")
        return get_with_ttl(key, r, ttl)

    remaining_ttl = r.ttl(key)
    if remaining_ttl < refresh_threshold and key not in _refresh_queue:
        _refresh_queue.add(key)
        t = threading.Thread(
            target=_background_refresh, args=(key, r, ttl), daemon=True
        )
        t.start()
        log.info(f"[STALE] {key} TTL={remaining_ttl}s — background refresh queued")

    return json.loads(cached)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Anti-pattern: lru_cache wrapping Redis — documented here for comparison
# ─────────────────────────────────────────────────────────────────────────────

# @lru_cache
# def anti_pattern_redis_get(key: str) -> str | None:
#     return r.get(key)
#
# WHY THIS BREAKS:
# - Redis gets updated   → lru_cache doesn't know
# - Redis invalidates    → lru_cache ignores it
# - New server starts    → empty lru_cache, stale Redis still "beats" it
# - Multi-worker         → each worker has its own out-of-sync lru_cache


# ─────────────────────────────────────────────────────────────────────────────
# Decision helper
# ─────────────────────────────────────────────────────────────────────────────

def choose_cache_strategy(
    multi_server: bool,
    needs_invalidation: bool,
    pure_computation: bool,
    high_traffic: bool,
) -> str:
    """Returns the recommended caching strategy based on requirements."""
    if pure_computation:
        return "lru_cache"
    if not multi_server and not needs_invalidation:
        return "TTLCache (cachetools)"
    if high_traffic:
        return "Redis + stampede lock"
    return "Redis with TTL"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    r = fakeredis.FakeRedis(decode_responses=True)
    key = "user:1"

    print("\n=== Cache Options Demo ===\n")

    print("-- Pattern 1: Redis with TTL --")
    reset_db_counter()
    get_with_ttl(key, r)
    get_with_ttl(key, r)
    print(f"   DB calls: {db_call_count()} (expected: 1)")

    print("\n-- Pattern 2: Redis + stampede lock --")
    r.delete(key)
    reset_db_counter()
    get_with_lock(key, r)
    get_with_lock(key, r)
    print(f"   DB calls: {db_call_count()} (expected: 1)")

    print("\n-- Pattern 3: TTLCache (single-process) --")
    _ttl_cache.clear()
    reset_db_counter()
    get_with_ttlcache(key)
    get_with_ttlcache(key)
    print(f"   DB calls: {db_call_count()} (expected: 1)")

    print("\n-- Pattern 4: Background refresh --")
    r.set(key, json.dumps({"name": "Alice", "balance": 1500.0}), ex=25)  # near expiry
    reset_db_counter()
    get_with_background_refresh(key, r)
    time.sleep(0.05)  # let background thread run
    print(f"   DB calls: {db_call_count()} (background refresh triggered)")

    print("\n-- Strategy selector --")
    print(f"   Multi-server user data: {choose_cache_strategy(True, True, False, False)}")
    print(f"   Pure formatting:        {choose_cache_strategy(False, False, True, False)}")
    print(f"   High traffic endpoint:  {choose_cache_strategy(True, True, False, True)}")


if __name__ == "__main__":
    main()
