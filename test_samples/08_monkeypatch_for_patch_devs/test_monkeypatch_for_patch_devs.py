"""
test_monkeypatch_for_patch_devs.py — Runnable demo for:
"Monkeypatch for a unittest.mock.patch Developer"

Every example shows the unittest.mock.patch version alongside
the pytest monkeypatch equivalent — so you can compare directly.

Run with:
    pytest -v test_monkeypatch_for_patch_devs.py
"""

import pytest
import os
from unittest.mock import patch, Mock

from module1 import Class1
import module2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic method patching — patch vs monkeypatch side by side
# ─────────────────────────────────────────────────────────────────────────────

def test_basic_with_patch():
    """unittest.mock.patch style: auto-creates mock, requires context manager."""
    with patch("module1.Class2.multiply", return_value=100) as mock_mul:
        obj = Class1()
        result = obj.compute(3, 5)
        assert result == 110
        mock_mul.assert_called_once_with(3, 5)


def test_basic_with_monkeypatch(monkeypatch):
    """
    pytest monkeypatch style: you supply the replacement directly.
    No mock object created automatically — just attribute reassignment.
    """
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 100)

    obj = Class1()
    result = obj.compute(3, 5)
    assert result == 110


# ─────────────────────────────────────────────────────────────────────────────
# 2. Replicating return_value= with monkeypatch
# ─────────────────────────────────────────────────────────────────────────────

def test_return_value_patch():
    """patch accepts return_value= directly."""
    with patch("module1.Class2.multiply", return_value=42):
        assert Class1().compute(1, 1) == 52


def test_return_value_monkeypatch(monkeypatch):
    """monkeypatch uses a lambda to set the return value."""
    monkeypatch.setattr("module1.Class2.multiply", lambda x, y: 42)
    assert Class1().compute(1, 1) == 52


# ─────────────────────────────────────────────────────────────────────────────
# 3. Replicating call assertions — combine monkeypatch with Mock
# ─────────────────────────────────────────────────────────────────────────────

def test_call_tracking_with_patch():
    """patch gives you a Mock with built-in call tracking."""
    with patch("module1.Class2.multiply") as mock_mul:
        mock_mul.return_value = 100
        Class1().compute(3, 5)
        mock_mul.assert_called_once_with(3, 5)


def test_call_tracking_with_monkeypatch(monkeypatch):
    """
    monkeypatch doesn't create mocks — supply your own Mock for tracking.
    monkeypatch handles lifecycle; Mock handles introspection.
    """
    mock_mul = Mock(return_value=100)
    monkeypatch.setattr("module1.Class2.multiply", mock_mul)

    Class1().compute(3, 5)

    mock_mul.assert_called_once_with(3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Instance method patching — self is required at class level
# ─────────────────────────────────────────────────────────────────────────────

def test_instance_method_with_patch():
    with patch.object(Class1, "compute_instance", return_value=999):
        assert Class1().compute_instance(10) == 999


def test_instance_method_with_monkeypatch(monkeypatch):
    """Patching at class level: lambda must accept self."""
    monkeypatch.setattr(Class1, "compute_instance", lambda self, v: 999)
    assert Class1().compute_instance(10) == 999


def test_specific_instance_with_monkeypatch(monkeypatch):
    """Patching a specific instance's bound method: no self required."""
    obj = Class1()
    monkeypatch.setattr(obj, "compute_instance", lambda v: 999)
    assert obj.compute_instance(10) == 999


# ─────────────────────────────────────────────────────────────────────────────
# 5. Environment variables — monkeypatch is cleaner here
# ─────────────────────────────────────────────────────────────────────────────

def test_env_var_with_patch():
    """patch.dict works but is more verbose."""
    with patch.dict(os.environ, {"API_KEY": "test-key-123"}):
        assert Class1().get_env() == "test-key-123"


def test_env_var_with_monkeypatch(monkeypatch):
    """monkeypatch.setenv is cleaner — purpose-built for env vars."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    assert Class1().get_env() == "test-key-123"


def test_env_var_deleted_with_monkeypatch(monkeypatch):
    """monkeypatch can also remove env vars cleanly."""
    monkeypatch.delenv("API_KEY", raising=False)
    assert Class1().get_env() == "missing"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Constructor patching — replace the class with a callable
# ─────────────────────────────────────────────────────────────────────────────

def test_constructor_with_patch():
    with patch("module1.Class2") as MockClass2:
        instance = Mock()
        instance.multiply.return_value = 50
        MockClass2.return_value = instance
        # Class2() now returns our fake instance


def test_constructor_with_monkeypatch(monkeypatch):
    """Replace the class itself with a lambda that returns a fake."""
    fake_instance = Mock()
    fake_instance.instance_double.return_value = 500
    monkeypatch.setattr("module1.Class2", lambda: fake_instance)

    obj = Class1()
    result = obj.compute_instance(10)
    assert result == 505  # 500 + 5


# ─────────────────────────────────────────────────────────────────────────────
# 7. Auto-cleanup demonstration
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_restored_after_context():
    """patch restores automatically when the with block exits."""
    original = module2.Class2.multiply
    with patch("module2.Class2.multiply", return_value=999):
        assert module2.Class2.multiply(2, 3) == 999
    # Restored — calling the original static-style method
    assert module2.Class2.multiply(2, 3) == 6


def test_monkeypatch_restored_after_test(monkeypatch):
    """monkeypatch restores automatically after the test function exits."""
    monkeypatch.setattr("module2.Class2.multiply", lambda x, y: 999)
    assert module2.Class2.multiply(2, 3) == 999
    # After this test, multiply is restored to the original


def test_original_still_intact_after_monkeypatch():
    """Verify the previous monkeypatch left no trace."""
    assert module2.Class2.multiply(2, 3) == 6


if __name__ == "__main__":
    print("Run with: pytest -v test_monkeypatch_for_patch_devs.py")
