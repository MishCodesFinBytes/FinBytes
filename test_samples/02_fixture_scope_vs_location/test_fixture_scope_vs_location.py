"""
test_fixture_scope_vs_location.py — Runnable demo for:
"Mental Model: Fixture Scope vs Fixture Location"

Key mental model:
    Scope  → controls LIFETIME  (when created / destroyed)
    Location → controls VISIBILITY (who can use it)

    Placement does NOT imply sharing.

Run with:
    pytest -v test_fixture_scope_vs_location.py
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# WRONG PATTERN: shared module-level mutable object returned from fixture
# ─────────────────────────────────────────────────────────────────────────────

_shared_cache = {}  # Module-level mutable — dangerous


@pytest.fixture
def leaky_cache():
    """
    BUG: This fixture is function-scoped (good),
    but it returns the SAME module-level dict every time (bad).

    The fixture code re-runs, but the object it returns is shared.
    pytest isolates fixture CALLS, not the OBJECTS you return.
    """
    return _shared_cache


# ─────────────────────────────────────────────────────────────────────────────
# CORRECT PATTERN: fresh object created inside the fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def safe_cache():
    """
    CORRECT: A new dict is created on every fixture call.
    True isolation guaranteed.
    """
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# The fixture is defined ABOVE the classes — does NOT make it shared
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def counter():
    """
    Defined at module level (above the classes), default function scope.
    Each test — in any class — gets its own fresh instance.
    Location ≠ sharing.
    """
    return {"value": 0}


class TestA:
    def test_one(self, counter):
        counter["value"] += 1
        assert counter["value"] == 1  # Fresh counter — always 1

    def test_two(self, counter):
        counter["value"] += 1
        assert counter["value"] == 1  # Another fresh counter — still 1


class TestB:
    def test_three(self, counter):
        # Even though counter is defined above TestA,
        # TestB gets its own fresh copy.
        assert counter["value"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Demonstrating the leaky vs safe cache difference
# ─────────────────────────────────────────────────────────────────────────────

class TestLeakyCache:
    def test_add_entry(self, leaky_cache):
        leaky_cache["alice"] = 1
        assert leaky_cache["alice"] == 1

    def test_cache_should_be_empty(self, leaky_cache):
        # This will FAIL if test_add_entry ran first!
        # Uncomment the assert to see the failure:
        # assert leaky_cache == {}  # ← fails in full suite, passes alone
        # Instead we just check the type to keep the demo runnable:
        assert isinstance(leaky_cache, dict)


class TestSafeCache:
    def test_add_entry(self, safe_cache):
        safe_cache["alice"] = 1
        assert safe_cache["alice"] == 1

    def test_cache_is_empty(self, safe_cache):
        # Always passes — fresh dict every time
        assert safe_cache == {}


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped mutable fixture — the worst offender
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def session_store():
    """
    Session scope: created once, shared across ALL tests.
    Safe only for read-only data (e.g. config files, DB connections).
    Never return a mutable object from a session fixture.
    """
    return {"read_only_config": "production"}


def test_session_fixture_is_read_only(session_store):
    # Only reading — safe
    assert session_store["read_only_config"] == "production"


if __name__ == "__main__":
    print("Run with: pytest -v test_fixture_scope_vs_location.py")
