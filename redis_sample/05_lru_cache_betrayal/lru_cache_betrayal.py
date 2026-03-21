"""
lru_cache_betrayal.py — Runnable demo for:
"The Drawer and the Fridge: Why lru_cache Can Betray You in Distributed Systems"

Demonstrates:
  - GOOD: lru_cache for pure deterministic functions (safe)
  - BAD:  lru_cache wrapping Redis calls (breaks invalidation)
  - The inconsistency bug in a simulated multi-worker environment
  - Correct alternative: Redis-only for shared mutable data

Tests (no Redis needed):
    pytest test_lru_cache_betrayal.py -v
"""

import json
import logging
import time
from functools import lru_cache
import fakeredis   # pip install fakeredis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Simulated shared Redis (one per "cluster", shared by all "workers")
# ─────────────────────────────────────────────────────────────────────────────

_shared_redis = fakeredis.FakeRedis(decode_responses=True)


def _set_balance(user_id: int, balance: float) -> None:
    """Update a user's balance in Redis (simulates a payment being processed)."""
    _shared_redis.set(f"balance:{user_id}", str(balance), ex=300)


def _get_raw_balance(user_id: int) -> str | None:
    return _shared_redis.get(f"balance:{user_id}")


# ─────────────────────────────────────────────────────────────────────────────
# GOOD: lru_cache for pure formatting — never changes for same input
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def format_currency(amount: float) -> str:
    """
    Pure function. lru_cache is safe here.
    format_currency(10.0) is always "£10.00" regardless of server, worker, or time.
    """
    return f"£{amount:,.2f}"


def get_formatted_balance(user_id: int) -> str:
    """GOOD pattern: Redis holds the balance; lru_cache only formats it."""
    raw = _get_raw_balance(user_id)
    amount = float(raw) if raw else 0.0
    return format_currency(amount)


# ─────────────────────────────────────────────────────────────────────────────
# BAD: lru_cache wrapping a Redis call — breaks correctness across workers
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def get_balance_bad(user_id: int) -> str | None:
    """
    ANTI-PATTERN: lru_cache on a Redis call.

    Problems:
    1. Redis gets updated → this cache never hears about it
    2. Different workers have different in-process caches
    3. Clearing Redis doesn't clear this cache
    4. Pub/Sub invalidation has no effect
    """
    return _shared_redis.get(f"balance:{user_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulated workers: each has its own process memory
# ─────────────────────────────────────────────────────────────────────────────

class Worker:
    """
    Simulates a Gunicorn worker process.
    In reality, each worker has completely isolated memory.
    Here we simulate that with separate per-instance caches.
    """

    def __init__(self, name: str):
        self.name = name
        self.log = logging.getLogger(name)
        self._local_cache: dict[str, str | None] = {}   # simulates per-worker lru_cache

    def get_balance_with_local_cache(self, user_id: int) -> str | None:
        """
        BAD pattern: worker caches the Redis value locally.
        If Redis is updated, this worker keeps serving the old value.
        """
        key = f"balance:{user_id}"
        if key in self._local_cache:
            self.log.info(f"[LOCAL HIT]  {key} → {self._local_cache[key]}")
            return self._local_cache[key]

        value = _shared_redis.get(key)
        self._local_cache[key] = value
        self.log.info(f"[LOCAL MISS] {key} → fetched from Redis: {value}")
        return value

    def get_balance_redis_only(self, user_id: int) -> str | None:
        """
        GOOD pattern: always read from Redis — consistent across all workers.
        """
        key = f"balance:{user_id}"
        value = _shared_redis.get(key)
        self.log.info(f"[REDIS]      {key} → {value}")
        return value


# ─────────────────────────────────────────────────────────────────────────────
# The inconsistency demo
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_inconsistency() -> dict:
    """
    Shows the bug:
      1. Worker A reads balance (misses local cache, fetches from Redis)
      2. Balance changes in Redis (e.g. a payment is processed)
      3. Worker A still serves the OLD value (local cache)
      4. Worker B, using Redis only, serves the CORRECT value
    """
    log = logging.getLogger("demo")
    worker_a = Worker("WorkerA")  # uses local cache (BAD)
    worker_b = Worker("WorkerB")  # uses Redis only (GOOD)

    # Initial balance
    _set_balance(1, 1000.00)
    log.info("Initial balance set to £1,000.00")

    # Both workers read — Worker A populates its local cache
    a_first = worker_a.get_balance_with_local_cache(1)
    b_first = worker_b.get_balance_redis_only(1)
    log.info(f"Before update: A={a_first}  B={b_first}")

    # Balance is updated in Redis (payment processed)
    _set_balance(1, 750.00)
    log.info("Balance updated in Redis to £750.00")

    # Worker A still serves old value; Worker B serves correct value
    a_after = worker_a.get_balance_with_local_cache(1)
    b_after = worker_b.get_balance_redis_only(1)
    log.info(f"After  update: A={a_after}  B={b_after}")

    inconsistent = a_after != b_after
    log.info(f"Inconsistency detected: {inconsistent}")
    return {
        "worker_a": a_after,
        "worker_b": b_after,
        "inconsistent": inconsistent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# When lru_cache IS acceptable
# ─────────────────────────────────────────────────────────────────────────────

import re

@lru_cache(maxsize=256)
def compile_pattern(pattern: str) -> re.Pattern:
    """Compiling regexes is pure and expensive — perfect for lru_cache."""
    return re.compile(pattern)


@lru_cache(maxsize=64)
def build_sql_template(table: str) -> str:
    """Building query templates is deterministic — safe for lru_cache."""
    return f"SELECT * FROM {table} WHERE id = %s"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== lru_cache Betrayal Demo ===\n")

    _set_balance(1, 1500.00)

    print("-- GOOD: pure function with lru_cache --")
    print(f"   {format_currency(1500.0)}")
    print(f"   {get_formatted_balance(1)}")

    print("\n-- BAD pattern demonstration --")
    result = demonstrate_inconsistency()
    print(f"\n   Worker A served: {result['worker_a']}")
    print(f"   Worker B served: {result['worker_b']}")
    print(f"   Users saw different answers: {result['inconsistent']}")

    print("\n-- Safe uses of lru_cache --")
    pattern = compile_pattern(r"\d{4}-\d{2}-\d{2}")
    print(f"   Compiled pattern: {pattern.pattern}")
    print(f"   SQL template: {build_sql_template('users')}")


if __name__ == "__main__":
    main()
