"""
fixed_311.py — Corrected versions of every pattern in broken_36.py.
Safe for Python 3.11 (and backwards compatible with 3.8+).

Each section is numbered to match the blog post checklist.
"""

from datetime import date, datetime
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Date/datetime arithmetic — convert explicitly
# ─────────────────────────────────────────────────────────────────────────────

def days_until(deadline: date) -> int:
    """Convert date → datetime before arithmetic to avoid TypeError."""
    today = datetime.now()
    deadline_dt = datetime.combine(deadline, datetime.min.time())
    delta = today - deadline_dt
    return delta.days


# ─────────────────────────────────────────────────────────────────────────────
# 2. Set ordering — sort explicitly before comparing
# ─────────────────────────────────────────────────────────────────────────────

def get_unique_tags(tags: list) -> list:
    """Return deduplicated tags in a deterministic order."""
    return sorted(set(tags))

# Safe assertion:
# assert get_unique_tags(["c", "a", "b"]) == ["a", "b", "c"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. None comparisons — guard before ordering
# ─────────────────────────────────────────────────────────────────────────────

def is_above_threshold(value, threshold=None) -> bool:
    """Guard against None before using ordering operators."""
    if threshold is None:
        return False
    return value > threshold


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exception handling — assert on type, not message text
# ─────────────────────────────────────────────────────────────────────────────

def parse_amount(s: str) -> int:
    return int(s)

# Safe test pattern (used in test_migration.py):
# with pytest.raises(ValueError):
#     parse_amount("abc")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Float equality — use math.isclose with tolerances
# ─────────────────────────────────────────────────────────────────────────────

def compound_rate(principal: float, rate: float, years: int) -> float:
    return principal * (1 + rate) ** years

def rates_are_equal(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Safe float comparison that survives precision differences across versions."""
    return math.isclose(a, b, rel_tol=rel_tol)

# Safe assertion:
# assert math.isclose(compound_rate(1000, 0.1, 3), 1331.0, rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. String/bytes — always specify encoding explicitly
# ─────────────────────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Explicit encoding prevents platform-dependent behaviour."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────────────
# 7. `is` vs `==` — always use == for value comparison
# ─────────────────────────────────────────────────────────────────────────────

def check_status(status: str) -> bool:
    """== checks value; is checks identity. Always use == for strings."""
    return status == "OK"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dict iteration order — safe patterns
# ─────────────────────────────────────────────────────────────────────────────

def get_config_keys(config: dict) -> list:
    """
    Dict insertion order is guaranteed from 3.7+, but do not rely on it
    for equality assertions — use sorted() for deterministic comparisons.
    """
    return sorted(config.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 9. Timing — poll instead of sleep
# ─────────────────────────────────────────────────────────────────────────────

import time


def wait_for_condition(condition_fn, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """
    Poll for a condition rather than using time.sleep() directly.
    Python 3.11 is faster — sleep-based tests become flaky.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition_fn():
            return True
        time.sleep(interval)
    return False
