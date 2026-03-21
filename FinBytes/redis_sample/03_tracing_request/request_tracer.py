"""
request_tracer.py — Runnable demo for:
"Tracing a User Request: From Browser to Database and Back"

Models each layer of the request stack and logs what happens at each hop:
  Browser → Load Balancer → Gunicorn Worker → Redis → Database → back

Run standalone:
    python request_tracer.py

Tests (no Redis needed):
    pytest test_request_tracer.py -v
"""

import json
import logging
import time
import uuid
import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Simulated DB
# ─────────────────────────────────────────────────────────────────────────────

_db = {
    1: {"user_id": 1, "name": "Alice", "balance": 1500.00},
    2: {"user_id": 2, "name": "Bob",   "balance": 275.50},
}

def db_fetch(user_id: int) -> dict | None:
    time.sleep(0.03)
    return _db.get(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Request context — carries a correlation ID through all layers
# ─────────────────────────────────────────────────────────────────────────────

class RequestContext:
    def __init__(self, user_id: int):
        self.request_id = str(uuid.uuid4())[:8]
        self.user_id = user_id
        self.timings: dict[str, float] = {}
        self.log = logging.getLogger(f"req:{self.request_id}")

    def mark(self, stage: str) -> None:
        self.timings[stage] = time.monotonic()

    def elapsed(self, from_stage: str, to_stage: str) -> str:
        diff = self.timings[to_stage] - self.timings[from_stage]
        return f"{diff * 1000:.1f}ms"


# ─────────────────────────────────────────────────────────────────────────────
# Each layer of the stack
# ─────────────────────────────────────────────────────────────────────────────

def browser_sends_request(user_id: int) -> RequestContext:
    """Step 1: Browser initiates an HTTP GET /profile/<user_id>"""
    ctx = RequestContext(user_id)
    ctx.mark("browser")
    ctx.log.info(f"→ GET /profile/{user_id} (request_id={ctx.request_id})")
    return ctx


def load_balancer(ctx: RequestContext) -> RequestContext:
    """Step 2: Load balancer routes to an available worker."""
    ctx.mark("lb")
    ctx.log.info(f"  [LB] routing request_id={ctx.request_id} to worker")
    return ctx


def gunicorn_worker(ctx: RequestContext, r: redis.Redis) -> dict | None:
    """
    Step 3–6: Worker checks Redis, falls back to DB if needed,
    writes result back to Redis, then returns to caller.
    """
    ctx.mark("worker_start")
    ctx.log.info(f"  [WORKER] handling request for user_id={ctx.user_id}")

    result = redis_layer(ctx, r)

    ctx.mark("worker_end")
    ctx.log.info(
        f"  [WORKER] done in {ctx.elapsed('worker_start', 'worker_end')}"
    )
    return result


def redis_layer(ctx: RequestContext, r: redis.Redis) -> dict | None:
    """Step 4: Check Redis cache."""
    key = f"user:{ctx.user_id}"
    ctx.mark("redis_check")

    cached = r.get(key)
    if cached is not None:
        ctx.mark("redis_hit")
        ctx.log.info(
            f"  [REDIS] CACHE HIT  key={key} "
            f"({ctx.elapsed('redis_check', 'redis_hit')})"
        )
        return json.loads(cached)

    ctx.log.info(f"  [REDIS] CACHE MISS key={key} — going to DB")
    return database_layer(ctx, r, key)


def database_layer(ctx: RequestContext, r: redis.Redis, key: str) -> dict | None:
    """Step 5: Fetch from DB on cache miss, then repopulate Redis."""
    ctx.mark("db_start")
    profile = db_fetch(ctx.user_id)
    ctx.mark("db_end")

    ctx.log.info(
        f"  [DB]    query took {ctx.elapsed('db_start', 'db_end')}"
    )

    if profile:
        r.set(key, json.dumps(profile), ex=300)
        ctx.log.info(f"  [REDIS] key={key} written back, TTL=300s")

    return profile


def send_response(ctx: RequestContext, profile: dict | None) -> dict:
    """Step 7: Format and return response to browser."""
    ctx.mark("response")
    total = ctx.elapsed("browser", "response")
    ctx.log.info(f"← response sent in {total} total")
    return {
        "request_id": ctx.request_id,
        "user_id": ctx.user_id,
        "profile": profile,
        "elapsed_ms": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full stack simulation
# ─────────────────────────────────────────────────────────────────────────────

def handle_request(user_id: int, r: redis.Redis) -> dict:
    """
    Simulates the full browser → LB → Worker → Redis → DB → response flow.
    Each layer is logged with its correlation ID.
    """
    ctx = browser_sends_request(user_id)
    ctx = load_balancer(ctx)
    profile = gunicorn_worker(ctx, r)
    return send_response(ctx, profile)


# ─────────────────────────────────────────────────────────────────────────────
# Developer-friendly Redis health checks
# ─────────────────────────────────────────────────────────────────────────────

def developer_checks(r: redis.Redis, user_id: int) -> None:
    """
    Checks a developer can safely run without sysadmin rights.
    """
    log = logging.getLogger("dev_checks")
    key = f"user:{user_id}"

    log.info(f"[CHECK] ping: {r.ping()}")
    log.info(f"[CHECK] get({key}): {r.get(key)}")
    log.info(f"[CHECK] ttl({key}): {r.ttl(key)}s")

    info = r.info("stats")
    hits = info.get("keyspace_hits", 0)
    misses = info.get("keyspace_misses", 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total > 0 else 0
    log.info(f"[CHECK] hit rate: {hit_rate:.1f}% ({hits} hits / {misses} misses)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    r = redis.Redis(host="localhost", port=6379, db=2, decode_responses=True)
    print("\n=== Request Tracing Demo ===\n")

    print("--- Request 1: Cold cache (MISS expected) ---")
    handle_request(1, r)

    print("\n--- Request 2: Warm cache (HIT expected) ---")
    handle_request(1, r)

    print("\n--- Developer health checks ---")
    developer_checks(r, 1)

    r.flushdb()


if __name__ == "__main__":
    main()
