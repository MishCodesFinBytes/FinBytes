"""
broken_36.py — Code patterns that worked (silently or by accident) in Python 3.6
but raise errors or produce wrong results in Python 3.11.

DO NOT run this file directly — it is intentionally broken.
It exists as a reference alongside fixed_311.py.

Each section is numbered to match the blog post checklist.
"""

from datetime import date, datetime
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. Date/datetime arithmetic — TypeError in 3.11
# ─────────────────────────────────────────────────────────────────────────────

def days_until(deadline: date) -> int:
    today = datetime.now()
    # BUG: mixing datetime and date — TypeError in 3.11
    delta = today - deadline
    return delta.days


# ─────────────────────────────────────────────────────────────────────────────
# 2. Set ordering assumption — may fail in 3.11
# ─────────────────────────────────────────────────────────────────────────────

def get_unique_tags(tags: list) -> list:
    # BUG: set iteration order is not guaranteed
    return list(set(tags))

# Fragile test that worked by accident in 3.6:
# assert get_unique_tags(["c", "a", "b"]) == ["c", "a", "b"]  # may fail


# ─────────────────────────────────────────────────────────────────────────────
# 3. Truthiness — None comparison with ordering operator
# ─────────────────────────────────────────────────────────────────────────────

def is_above_threshold(value, threshold=None):
    # BUG: TypeError in 3.11 if threshold is None
    if value > threshold:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exception message assertion — brittle test pattern
# ─────────────────────────────────────────────────────────────────────────────

def parse_amount(s: str) -> int:
    return int(s)

# Fragile test:
# try:
#     parse_amount("abc")
# except ValueError as e:
#     assert "invalid literal" in str(e)   # message changed in 3.11


# ─────────────────────────────────────────────────────────────────────────────
# 5. Float equality — precision differences surface in 3.11
# ─────────────────────────────────────────────────────────────────────────────

def compound_rate(principal: float, rate: float, years: int) -> float:
    return principal * (1 + rate) ** years

# Fragile assertion:
# assert compound_rate(1000, 0.1, 3) == 1331.0   # may fail due to float precision


# ─────────────────────────────────────────────────────────────────────────────
# 6. String/bytes mixing — implicit encoding assumption
# ─────────────────────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    # BUG: no encoding specified — platform-dependent, may break in 3.11 on some systems
    with open(path) as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# 7. `is` for string equality — worked by accident via interning in 3.6
# ─────────────────────────────────────────────────────────────────────────────

def check_status(status: str) -> bool:
    # BUG: `is` checks identity, not value — unreliable in 3.11
    if status is "OK":
        return True
    return False
