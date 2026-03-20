"""
conftest.py — Demonstrates the leaky fixture anti-pattern described in:
"When Pytest Tests Pass — and Still Break the Enterprise Build"

The WRONG version (module-scoped mutable fixture) is shown commented out.
The CORRECT version (function-scoped) is active.

Run with:
    pytest -v
"""

import pytest


# ─────────────────────────────────────────────
# WRONG: module-scoped mutable fixture
# Uncomment this block and comment out the correct version to see CI-style failure
# ─────────────────────────────────────────────
# @pytest.fixture(scope="module")
# def counter():
#     return {"value": 0}


# ─────────────────────────────────────────────
# CORRECT: function-scoped — each test gets a fresh dict
# ─────────────────────────────────────────────
@pytest.fixture(scope="function")
def counter():
    return {"value": 0}
