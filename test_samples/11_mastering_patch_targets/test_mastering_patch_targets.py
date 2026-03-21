"""
test_mastering_patch_targets.py — Runnable demo for:
"Mastering Python Patch Targets: Top-Level Functions, Instance Methods,
 and Static Methods"

Complete reference for every patch target type:
  top-level function, instance method, static method, class method,
  multiple classes with the same method name.

Run with:
    pytest -v test_mastering_patch_targets.py
"""

import pytest
from unittest.mock import patch, Mock

from service import compute, CalculatorA, CalculatorB


# ─────────────────────────────────────────────────────────────────────────────
# 1. Top-level function
#    Patch target: "service.compute"
# ─────────────────────────────────────────────────────────────────────────────

@patch("service.compute")
def test_patch_top_level_function(mock_compute):
    mock_compute.return_value = 99
    assert compute(5) == 99
    # Class methods are completely unaffected
    assert CalculatorA().compute(5) == 6
    assert CalculatorB().compute(5) == 15


def test_patch_top_level_with_monkeypatch(monkeypatch):
    monkeypatch.setattr("service.compute", lambda x: 99)
    assert compute(5) == 99
    assert CalculatorA().compute(5) == 6  # untouched


# ─────────────────────────────────────────────────────────────────────────────
# 2. Instance method — CalculatorA
#    Patch target: "service.CalculatorA.compute"
# ─────────────────────────────────────────────────────────────────────────────

@patch("service.CalculatorA.compute")
def test_patch_calculator_a_instance_method(mock_method):
    mock_method.return_value = 100
    assert CalculatorA().compute(5) == 100
    # Other methods unaffected
    assert CalculatorB().compute(5) == 15
    assert compute(5) == 10


def test_patch_calculator_a_with_monkeypatch(monkeypatch):
    monkeypatch.setattr("service.CalculatorA.compute", lambda self, x: 100)
    assert CalculatorA().compute(5) == 100
    assert CalculatorB().compute(5) == 15  # unaffected


# ─────────────────────────────────────────────────────────────────────────────
# 3. Instance method — CalculatorB (same name "compute", different class)
#    Patch target: "service.CalculatorB.compute"
# ─────────────────────────────────────────────────────────────────────────────

@patch("service.CalculatorB.compute")
def test_patch_calculator_b_instance_method(mock_method):
    mock_method.return_value = 200
    assert CalculatorB().compute(5) == 200
    # CalculatorA completely unaffected
    assert CalculatorA().compute(5) == 6
    assert compute(5) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 4. Static method
#    Patch target: "service.CalculatorA.compute_static"
# ─────────────────────────────────────────────────────────────────────────────

@patch("service.CalculatorA.compute_static")
def test_patch_static_method(mock_static):
    mock_static.return_value = 999
    assert CalculatorA.compute_static(5) == 999
    # Instance and top-level unaffected
    assert CalculatorA().compute(5) == 6
    assert compute(5) == 10


def test_patch_static_with_monkeypatch(monkeypatch):
    """Static methods don't receive self or cls — plain lambda."""
    monkeypatch.setattr("service.CalculatorA.compute_static", lambda x: 999)
    assert CalculatorA.compute_static(5) == 999


# ─────────────────────────────────────────────────────────────────────────────
# 5. Class method
#    Patch target: "service.CalculatorA.compute_class"
# ─────────────────────────────────────────────────────────────────────────────

@patch("service.CalculatorA.compute_class")
def test_patch_class_method(mock_class):
    mock_class.return_value = 555
    assert CalculatorA.compute_class(5) == 555
    # Nothing else affected
    assert CalculatorA().compute(5) == 6
    assert compute(5) == 10


def test_patch_class_method_with_monkeypatch(monkeypatch):
    """Class methods receive cls as first argument."""
    monkeypatch.setattr("service.CalculatorA.compute_class", lambda cls, x: x + 1000)
    assert CalculatorA.compute_class(5) == 1005


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prove patches are scoped — patching A does not affect B
# ─────────────────────────────────────────────────────────────────────────────

def test_patches_are_scoped_per_class(monkeypatch):
    """
    A.compute and B.compute share the same name but are independent targets.
    Patching one never affects the other.
    """
    monkeypatch.setattr("service.CalculatorA.compute", lambda self, x: 777)

    assert CalculatorA().compute(5) == 777   # patched
    assert CalculatorB().compute(5) == 15    # real
    assert compute(5) == 10                  # real


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bonus: live demo of all real values before any patching
# ─────────────────────────────────────────────────────────────────────────────

def test_all_real_values():
    """Baseline: verify the real implementations before any patching."""
    assert compute(5) == 10
    assert CalculatorA().compute(5) == 6
    assert CalculatorB().compute(5) == 15
    assert CalculatorA.compute_static(5) == 50
    assert CalculatorA.compute_class(5) == 500


# ─────────────────────────────────────────────────────────────────────────────
# 8. Quick reference summary (no assertions — documentation test)
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_target_reference():
    """
    Patch target quick reference:

    | Code                              | Patch target                          |
    |-----------------------------------|---------------------------------------|
    | compute(x)                        | service.compute                       |
    | CalculatorA().compute(x)          | service.CalculatorA.compute           |
    | CalculatorB().compute(x)          | service.CalculatorB.compute           |
    | CalculatorA.compute_static(x)     | service.CalculatorA.compute_static    |
    | CalculatorA.compute_class(x)      | service.CalculatorA.compute_class     |

    Rule: always patch where Python looks up the object, not where it was defined.
    """
    pass  # This is a documentation test — read the docstring


if __name__ == "__main__":
    print("Run with: pytest -v test_mastering_patch_targets.py")
