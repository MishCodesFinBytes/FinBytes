"""
test_monkeypatching_correct_target.py — Runnable demo for:
"Monkeypatching the Correct Target in Pytest"

Demonstrates the exact patch target for each import style:
  - direct import (from x import y)  → patch the consumer module
  - module import (import x; x.y())  → patch the source module

Run with:
    pytest -v test_monkeypatching_correct_target.py
"""

import pytest
import processor
import services


# ─────────────────────────────────────────────────────────────────────────────
# 1. Top-level function — direct import
#    processor does: from services import top_level_function
#    → patch processor.top_level_function (NOT services.top_level_function)
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_top_level_function(monkeypatch):
    monkeypatch.setattr("processor.top_level_function", lambda: "mocked top")
    result = processor.run_all()
    assert result[0] == "mocked top"


def test_patching_services_top_level_has_no_effect(monkeypatch):
    """processor already holds its own reference — patching the source is too late."""
    monkeypatch.setattr("services.top_level_function", lambda: "should not appear")
    result = processor.run_all()
    assert result[0] == "real top level"  # patch had no effect


# ─────────────────────────────────────────────────────────────────────────────
# 2. Instance method on directly imported class
#    processor does: from services import A
#    → patch processor.A.method
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_instance_method(monkeypatch):
    monkeypatch.setattr("processor.A.method", lambda self: "mocked A.method")
    result = processor.run_all()
    assert result[1] == "mocked A.method"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Static method — no self required
#    processor does: from services import A
#    → patch processor.A.static_method
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_static_method(monkeypatch):
    monkeypatch.setattr("processor.A.static_method", lambda: "mocked static")
    result = processor.run_all()
    assert result[2] == "mocked static"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Class accessed via module reference (import services; services.B())
#    → patch services.B.method (not processor.B — B was never imported directly)
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_module_ref_class(monkeypatch):
    monkeypatch.setattr("services.B.method", lambda self: "mocked B.method")
    result = processor.run_all()
    assert result[3] == "mocked B.method"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Helper function declared in services (above class A)
#    A.call_helper() calls helper() which lives in services namespace
#    → patch services.helper (even though A was imported into processor)
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_helper_in_services(monkeypatch):
    monkeypatch.setattr("services.helper", lambda: "mocked helper")
    result = processor.run_all()
    assert result[4] == "mocked helper"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full run — all patches applied together
# ─────────────────────────────────────────────────────────────────────────────

def test_run_all_fully_patched(monkeypatch):
    monkeypatch.setattr("processor.top_level_function", lambda: "mocked top")
    monkeypatch.setattr("processor.A.method", lambda self: "mocked A.method")
    monkeypatch.setattr("processor.A.static_method", lambda: "mocked static")
    monkeypatch.setattr("services.B.method", lambda self: "mocked B.method")
    monkeypatch.setattr("services.helper", lambda: "mocked helper")

    result = processor.run_all()

    assert result == [
        "mocked top",
        "mocked A.method",
        "mocked static",
        "mocked B.method",
        "mocked helper",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Two classes with the same method name — patches are independent
# ─────────────────────────────────────────────────────────────────────────────

def test_same_method_name_different_classes(monkeypatch):
    """Patching A.method does not affect B.method and vice versa."""
    monkeypatch.setattr("processor.A.method", lambda self: "only A patched")
    # B.method is NOT patched

    assert services.A().method() == "only A patched"
    assert services.B().method() == "real B.method"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Debugging trick: inspect the actual binding
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_binding():
    """
    processor.A and services.A should be the same object (direct import).
    If they were different, you'd need to patch both.
    """
    assert processor.A is services.A


if __name__ == "__main__":
    print("Run with: pytest -v test_monkeypatching_correct_target.py")
