"""
chat_app.py — Core Redis chat application used across all four demo posts:
  - Learning & Troubleshooting
  - More Troubleshooting
  - Enhanced Logging Demo
  - Multiple Workers & Race Conditions
  - Live Session Simulation

Features:
  - Redis Pub/Sub (real-time delivery)
  - Redis list cache (history with TTL + LTRIM eviction)
  - Atomic pipeline (LPUSH + LTRIM + EXPIRE)
  - Two listener threads simulating multiple workers
  - Detailed timestamped logging for every stage
  - Graceful Redis connection error handling

Run with a local Redis:
    redis-server
    python chat_app.py

Then hit the endpoints:
    POST http://127.0.0.1:5000/chat    {"message": "Hello!"}
    GET  http://127.0.0.1:5000/history
    GET  http://127.0.0.1:5000/ping

Tests (no Redis needed):
    pip install fakeredis flask
    pytest test_chat_app.py -v
"""

import threading
import time
import logging

import redis
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)


def log(msg: str) -> None:
    logging.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Redis connection factory (injectable for testing)
# ─────────────────────────────────────────────────────────────────────────────

def make_redis(host: str = "localhost", port: int = 6379) -> redis.Redis:
    return redis.Redis(host=host, port=port, db=0, decode_responses=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat core
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL = "global_chat"
MESSAGE_LIST_KEY = "chat_messages"
CACHE_TTL = 10        # Short TTL for demo — expiry visible in seconds
MAX_MESSAGES = 10     # LTRIM keeps only the last N messages


def start_listener(r: redis.Redis, name: str = "Listener") -> None:
    """
    Starts a Pub/Sub listener in the current thread.
    Call this in a daemon thread.
    """
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNEL)
    log(f"{name} started listening...")
    for message in pubsub.listen():
        if message["type"] == "message":
            log(f"{name} received: {message['data']}")


def send_message(r: redis.Redis, message: str) -> bool:
    """
    Publishes message to Pub/Sub AND appends it to the Redis history list.
    Uses an atomic pipeline so LPUSH + LTRIM + EXPIRE are one operation.
    Returns True on success, False if Redis is unavailable.
    """
    try:
        # Pub/Sub: live delivery to all current subscribers
        r.publish(CHANNEL, message)
        log(f"[PUB/SUB SENT] {message}")

        # Cache: persist last N messages with TTL
        pipe = r.pipeline()
        pipe.lpush(MESSAGE_LIST_KEY, message)
        pipe.ltrim(MESSAGE_LIST_KEY, 0, MAX_MESSAGES - 1)
        pipe.expire(MESSAGE_LIST_KEY, CACHE_TTL)
        pipe.execute()
        log(f"[CACHE] Message cached (TTL={CACHE_TTL}s)")
        return True

    except redis.exceptions.ConnectionError:
        log("[ERROR] Redis not available! Message not sent.")
        return False


def get_history(r: redis.Redis) -> list[str]:
    """
    Returns cached chat history from Redis list.
    Logs cache hit or miss.
    """
    try:
        messages = r.lrange(MESSAGE_LIST_KEY, 0, -1)
        if messages:
            log(f"[CACHE HIT] Retrieved {len(messages)} messages")
        else:
            log("[CACHE MISS] No messages in cache")
        return messages
    except redis.exceptions.ConnectionError:
        log("[ERROR] Redis not available! Cannot retrieve history.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Flask app factory (accepts an injectable Redis for testing)
# ─────────────────────────────────────────────────────────────────────────────

def create_app(r: redis.Redis) -> Flask:
    """
    Returns a configured Flask app using the provided Redis connection.
    Using a factory makes the app fully testable without monkeypatching globals.
    """
    app = Flask(__name__)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"status": "failed", "error": "No message provided"}), 400
        success = send_message(r, message)
        return jsonify({"status": "sent" if success else "failed"})

    @app.route("/history")
    def history():
        messages = get_history(r)
        return jsonify({"messages": messages})

    @app.route("/ping")
    def ping():
        return jsonify({"status": "running"})

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — starts two listener threads then runs Flask
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r = make_redis()

    # Two listener threads simulate multiple Pub/Sub workers
    for name in ["Listener1", "Listener2"]:
        t = threading.Thread(target=start_listener, args=(r, name), daemon=True)
        t.start()

    log("Starting multi-worker chat demo...")
    app = create_app(r)
    app.run(debug=True, use_reloader=False)
