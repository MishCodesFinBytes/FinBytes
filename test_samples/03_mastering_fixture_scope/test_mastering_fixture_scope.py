"""
test_mastering_fixture_scope.py — Runnable demo for:
"Mastering pytest.fixture Scope: Practical Usage, Variations, and Gotchas"

Covers all five scopes: function, class, module, package, session.
Includes: yield teardown, autouse, parameterized, scope dependency rules.

Run with:
    pytest -v test_mastering_fixture_scope.py
"""

import pytest
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNCTION SCOPE (default — safest)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def function_user():
    """Fresh for every single test. The safest default."""
    return {"name": "Alice", "score": 0}


def test_function_scope_isolated_a(function_user):
    function_user["score"] += 10
    assert function_user["score"] == 10


def test_function_scope_isolated_b(function_user):
    # Still 0 — completely isolated from the test above
    assert function_user["score"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASS SCOPE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def class_user():
    """Shared across all tests in a class. Risky if mutable."""
    return {"name": "Bob", "calls": 0}


class TestClassScope:
    def test_first_call(self, class_user):
        class_user["calls"] += 1
        assert class_user["calls"] == 1

    def test_second_call(self, class_user):
        # Shared with test_first_call — calls is already 1
        class_user["calls"] += 1
        assert class_user["calls"] == 2  # ← demonstrates sharing


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODULE SCOPE
# ─────────────────────────────────────────────────────────────────────────────

_module_setup_count = {"n": 0}


@pytest.fixture(scope="module")
def module_connection():
    """
    Created once per file. Simulates an expensive resource like a DB engine.
    Use for read-only shared resources — reset state explicitly if needed.
    """
    _module_setup_count["n"] += 1
    return {"engine": "FakeDB", "setup_count": _module_setup_count["n"]}


def test_module_scope_a(module_connection):
    assert module_connection["engine"] == "FakeDB"
    assert module_connection["setup_count"] == 1  # Created once


def test_module_scope_b(module_connection):
    # Same object as test_module_scope_a
    assert module_connection["setup_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. YIELD FIXTURE with teardown
# ─────────────────────────────────────────────────────────────────────────────

_teardown_log = []


@pytest.fixture(scope="function")
def temp_resource():
    """
    Yield fixtures run teardown code after the test completes.
    The cleanup runs no matter what — even if the test fails.
    """
    resource = {"id": "temp-001", "active": True}
    yield resource
    # Teardown
    resource["active"] = False
    _teardown_log.append("cleaned up")


def test_yield_fixture(temp_resource):
    assert temp_resource["active"] is True
    # After this test, teardown runs automatically


def test_yield_teardown_ran():
    # By this point, the previous fixture has been torn down
    assert "cleaned up" in _teardown_log


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTOUSE FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """
    Runs automatically for every test in scope.
    Use sparingly — obscures dependencies and makes tests harder to understand.
    """
    monkeypatch.setenv("APP_ENV", "test")


def test_autouse_env_is_set():
    assert os.environ["APP_ENV"] == "test"


def test_autouse_env_also_set_here():
    # autouse fires again — independent env patch
    assert os.environ["APP_ENV"] == "test"


# ─────────────────────────────────────────────────────────────────────────────
# 6. PARAMETERIZED FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[1, 2, 3])
def number(request):
    """
    Runs the consuming test once per param value.
    3 params × 1 test = 3 test executions.
    """
    return request.param


def test_number_is_positive(number):
    assert number > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. SCOPE DEPENDENCY RULE — wider cannot depend on narrower
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def session_config():
    return {"env": "ci"}


@pytest.fixture(scope="function")
def function_with_session_dep(session_config):
    """
    VALID: function scope can depend on session scope (wider → narrower is fine).
    """
    return {"config": session_config, "extra": "per-test"}


def test_scope_dependency(function_with_session_dep):
    assert function_with_session_dep["config"]["env"] == "ci"
    assert function_with_session_dep["extra"] == "per-test"


# ─────────────────────────────────────────────────────────────────────────────
# 8. SESSION-SCOPED MUTABLE ANTI-PATTERN (shown, not used)
# ─────────────────────────────────────────────────────────────────────────────

# @pytest.fixture(scope="session")
# def global_cache():
#     return {}   ← ANTI-PATTERN: mutable session fixture leaks across all tests


if __name__ == "__main__":
    print("Run with: pytest -v test_mastering_fixture_scope.py")
