"""
test_monkeypatch_edge_cases.py — Runnable demo for:
"Monkeypatch Edge Cases That Break Senior Engineers"

Each test isolates a specific edge case with a WRONG and CORRECT approach.

Run with:
    pytest -v test_monkeypatch_edge_cases.py
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

import module2
from module1 import Service


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 1: Patching the wrong import namespace
# ─────────────────────────────────────────────────────────────────────────────

def test_correct_namespace_patch(monkeypatch):
    """
    module1 does `from module2 import Class2` — it holds its own reference.
    Patching module2.Class2.do_work has NO effect on module1.
    Must patch module1.Class2.do_work.
    """
    monkeypatch.setattr("module1.Class2.do_work", lambda self: "patched correctly")

    svc = Service()
    assert svc.run() == "patched correctly"


def test_wrong_namespace_has_no_effect(monkeypatch):
    """
    Patching the definition site (module2) when module1 already imported Class2
    leaves module1's reference unchanged.
    """
    monkeypatch.setattr("module2.Class2.do_work", lambda self: "should not appear")

    svc = Service()
    # module1 still uses its own reference — patch had no effect
    assert svc.run() == "real work"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 2: Patching after object instantiation (too late)
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_before_instantiation(monkeypatch):
    """
    CORRECT: patch ExternalClient BEFORE Service() is created,
    so __init__ picks up the fake.
    """
    fake_client = Mock()
    fake_client.connect.return_value = "fake connection"
    monkeypatch.setattr("module1.ExternalClient", lambda: fake_client)

    svc = Service()
    assert svc.use_client() == "fake connection"


def test_patch_after_instantiation_has_no_effect(monkeypatch):
    """
    WRONG ORDER: Service() already stored a real ExternalClient in __init__.
    Patching ExternalClient afterward does nothing to the existing instance.
    """
    svc = Service()  # real ExternalClient bound here

    fake_client = Mock()
    fake_client.connect.return_value = "fake connection"
    monkeypatch.setattr("module1.ExternalClient", lambda: fake_client)

    # use_client() still calls the real client bound at construction
    assert svc.use_client() == "real connection"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 3: Instance method requires `self`
# ─────────────────────────────────────────────────────────────────────────────

def test_instance_method_needs_self(monkeypatch):
    """When patching a class-level instance method, include self."""
    monkeypatch.setattr(
        "module1.Class2.do_work",
        lambda self: "mocked with self",
    )
    assert Service().run() == "mocked with self"


def test_patching_specific_instance_no_self_needed(monkeypatch):
    """Patching a bound method on a specific instance — no self required."""
    svc = Service()
    monkeypatch.setattr(svc.client, "connect", lambda: "instance-level patch")
    assert svc.use_client() == "instance-level patch"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 4: Static method — no self, no cls
# ─────────────────────────────────────────────────────────────────────────────

def test_static_method_patch(monkeypatch):
    """Static methods don't receive self or cls — patch with plain lambda."""
    monkeypatch.setattr("module2.Class2.static_op", lambda: "mocked static")
    assert module2.Class2.static_op() == "mocked static"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 5: Class method — must include cls
# ─────────────────────────────────────────────────────────────────────────────

def test_classmethod_patch(monkeypatch):
    """Class methods receive cls as first argument."""
    monkeypatch.setattr("module2.Class2.class_op", lambda cls, x: x + 1000)
    assert module2.Class2.class_op(5) == 1005


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 6: Property — must wrap with property()
# ─────────────────────────────────────────────────────────────────────────────

def test_property_patch(monkeypatch):
    """Properties are descriptors. Must remain descriptors after patching."""
    monkeypatch.setattr(
        module2.Class2,
        "label",
        property(lambda self: "mocked label"),
    )
    obj = module2.Class2()
    assert obj.label == "mocked label"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 7: Mutable shared state — use setitem, not setattr
# ─────────────────────────────────────────────────────────────────────────────

def test_mutable_config_via_setitem(monkeypatch):
    """
    Monkeypatch restores attributes, but not mutations to mutable objects.
    Use monkeypatch.setitem() to safely patch dict entries.
    """
    monkeypatch.setitem(module2.CONFIG, "timeout", 1)

    svc = Service()
    assert svc.use_config() == 1
    # After test, CONFIG["timeout"] is automatically restored to 30


def test_config_restored_after_setitem():
    """Verify setitem restored the original value."""
    assert module2.CONFIG["timeout"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 8: Async functions — must patch with async def or AsyncMock
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_function_patch(monkeypatch):
    """
    Patching an async function with a regular lambda raises TypeError
    because the result isn't awaitable. Use AsyncMock or async def.
    """
    monkeypatch.setattr(
        "module2.async_fetch",
        AsyncMock(return_value="mocked async result"),
    )
    result = await module2.async_fetch()
    assert result == "mocked async result"


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 9: Verify patch actually executed (silent success trap)
# ─────────────────────────────────────────────────────────────────────────────

def test_patch_was_actually_called(monkeypatch):
    """
    Trust but verify: combine monkeypatch with Mock to confirm
    the patched code path was actually reached.
    """
    mock_fn = Mock(return_value="verified")
    monkeypatch.setattr("module1.Class2.do_work", mock_fn)

    svc = Service()
    result = svc.run()

    assert result == "verified"
    mock_fn.assert_called_once()


if __name__ == "__main__":
    print("Run with: pytest -v test_monkeypatch_edge_cases.py")
