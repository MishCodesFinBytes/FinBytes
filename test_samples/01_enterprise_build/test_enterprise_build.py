"""
test_enterprise_build.py — Runnable demo for:
"When Pytest Tests Pass — and Still Break the Enterprise Build"

Illustrates how a module-scoped mutable fixture causes order-dependent
test failures in CI, while passing locally when tests run in isolation.

Also includes the defensive CI guard that detects leaked sys.modules mocks.

Run with:
    pytest -v test_enterprise_build.py
"""

import sys
import pytest
from unittest.mock import Mock


# ─────────────────────────────────────────────────────────────────────────────
# The leaky fixture scenario
# ─────────────────────────────────────────────────────────────────────────────

class TestOne:
    """These tests increment the shared counter."""

    def test_increment(self, counter):
        counter["value"] += 1
        assert counter["value"] == 1

    def test_increment_again(self, counter):
        # With function scope: counter starts at 0 again — passes.
        # With module scope: counter carries 1 from test_increment — fails.
        counter["value"] += 1
        assert counter["value"] == 1


class TestTwo:
    """These tests assume the counter starts at zero."""

    def test_starts_at_zero(self, counter):
        # With function scope: always 0 — passes.
        # With module scope: if TestOne ran first, this is already 1 — fails.
        assert counter["value"] == 0

    def test_reset_behaviour(self, counter):
        counter["value"] = 42
        assert counter["value"] == 42


# ─────────────────────────────────────────────────────────────────────────────
# Defensive CI guard: detect leaked sys.modules mocks
# ─────────────────────────────────────────────────────────────────────────────

def test_no_mocked_modules_leaked():
    """
    A low-cost guardrail that catches the real enterprise incident:
    a fixture that did `sys.modules["some_dependency"] = Mock()`
    without ever cleaning up.

    This test would have caught the original CI failure immediately.
    """
    leaked = [
        name for name, mod in sys.modules.items()
        if isinstance(mod, Mock)
    ]
    assert not leaked, f"Mocked modules leaked into sys.modules: {leaked}"


# ─────────────────────────────────────────────────────────────────────────────
# Demo: what a leaky sys.modules mock looks like (isolated, self-cleaning)
# ─────────────────────────────────────────────────────────────────────────────

def test_simulate_leaky_sys_modules_mock(monkeypatch):
    """
    Simulates the kind of fixture that caused the enterprise CI incident.
    monkeypatch automatically restores sys.modules after the test — which
    is exactly why you should use monkeypatch instead of mutating directly.
    """
    fake_module = Mock()
    monkeypatch.setitem(sys.modules, "some_fake_dependency", fake_module)

    # Within this test, the fake module is available
    import importlib
    assert isinstance(sys.modules.get("some_fake_dependency"), Mock)

    # After the test, monkeypatch restores sys.modules automatically.
    # No leak. No CI chaos.


if __name__ == "__main__":
    # Quick manual demo
    print("Run with: pytest -v test_enterprise_build.py")
