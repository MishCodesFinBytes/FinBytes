"""
test_monkeypatch_field_guide.py — Runnable demo for:
"pytest Monkeypatch: A Practical Field Guide (with Inline Return Patterns)"

Covers every inline return pattern from the reference table:
  constant return, input-dependent, conditional, exception,
  single instance, retry simulation, multiple patches.

Run with:
    pytest -v test_monkeypatch_field_guide.py
"""

import pytest
from unittest.mock import Mock

from module1 import Class1
from module2 import Class2


# ─────────────────────────────────────────────────────────────────────────────
# Core rule: patch where the symbol is looked up, not where defined
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_correct_namespace(monkeypatch):
    """module1 imported Class2 directly — patch module1.Class2, not module2."""
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 100)
    monkeypatch.setattr("module1.Class2.add", lambda x, y: 50)

    obj = Class1()
    assert obj.compute_library(3, 5) == 150


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1: Constant return
# ─────────────────────────────────────────────────────────────────────────────

def test_constant_return(monkeypatch):
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 42)
    assert Class1().compute_library(1, 1) == 42 + Class2.add(1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2: Instance method (self required at class level)
# ─────────────────────────────────────────────────────────────────────────────

def test_instance_method_with_self(monkeypatch):
    monkeypatch.setattr(
        "module1.Class2.instance_double",
        lambda self, value: 999,
    )
    assert Class1().compute_instance(10) == 1004  # 999 + 5


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3: Patch only a single object instance (no self needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_single_instance_patch(monkeypatch):
    obj = Class1()
    monkeypatch.setattr(obj.helper, "instance_double", lambda value: 500)
    assert obj.compute_instance(10) == 505


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 4: Input-dependent return
# ─────────────────────────────────────────────────────────────────────────────

def test_input_dependent_return(monkeypatch):
    """Simulate alternate logic based on input values."""
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: x + y)
    monkeypatch.setattr("module1.Class2.add", lambda x, y: 0)

    result = Class1().compute_library(3, 4)
    assert result == 7  # (3+4) + 0


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 5: Conditional branch (edge case testing)
# ─────────────────────────────────────────────────────────────────────────────

def test_conditional_branch(monkeypatch):
    """Simulate negative input guard."""
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 0 if x < 0 else 1)

    obj = Class1()
    # With positive input
    monkeypatch.setattr("module1.Class2.add", lambda x, y: 0)
    assert obj.compute_library(5, 2) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 6: Force an exception (test failure paths)
# ─────────────────────────────────────────────────────────────────────────────

def test_raise_exception(monkeypatch):
    """Use a named function for readability when raising exceptions."""
    def fail_multiply(*args, **kwargs):
        raise ValueError("Database unavailable")

    monkeypatch.setattr("module1.Class2.multiply", fail_multiply)

    with pytest.raises(ValueError, match="Database unavailable"):
        Class1().compute_library(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 7: Stateful patch — retry simulation
# ─────────────────────────────────────────────────────────────────────────────

def test_retry_simulation(monkeypatch):
    """
    First call raises, second call succeeds.
    Useful for testing retry logic in services.
    """
    calls = {"n": 0}

    def flaky_multiply(x, y):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient failure")
        return x * y

    monkeypatch.setattr("module1.Class2.multiply", flaky_multiply)

    # First call should fail
    monkeypatch.setattr("module1.Class2.add", lambda x, y: 0)
    with pytest.raises(ConnectionError):
        Class1().compute_library(3, 5)

    # Second call should succeed
    assert Class1().compute_library(3, 5) == 15


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 8: Multiple patches in one test
# ─────────────────────────────────────────────────────────────────────────────

def test_multiple_patches(monkeypatch):
    """Patch several methods simultaneously to compose a test scenario."""
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 10)
    monkeypatch.setattr("module1.Class2.add", lambda x, y: 5)

    result = Class1().compute_library(99, 99)
    assert result == 15


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 9: Combine monkeypatch + Mock for call verification
# ─────────────────────────────────────────────────────────────────────────────

def test_mock_with_monkeypatch_for_call_tracking(monkeypatch):
    """
    Monkeypatch handles cleanup; Mock handles introspection.
    Use this when you need both patching AND call assertions.
    """
    mock_multiply = Mock(return_value=7)
    monkeypatch.setattr("module1.Class2.multiply", mock_multiply)
    monkeypatch.setattr("module1.Class2.add", lambda x, y: 0)

    Class1().compute_library(3, 5)

    mock_multiply.assert_called_once_with(3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 10: Auto-cleanup — no teardown needed
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_is_temporary(monkeypatch):
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 999)
    assert Class1().compute_library(1, 0) == 999 + Class2.add(1, 0)


def test_original_restored():
    """After the previous test, multiply is back to real behaviour."""
    assert Class2.multiply(3, 4) == 12


if __name__ == "__main__":
    print("Run with: pytest -v test_monkeypatch_field_guide.py")
