"""
test_email_sending.py — Runnable demo for:
"Testing Email Sending in Python: Verifying Call Count, Order,
 and Arguments with monkeypatch and mock.call"

Demonstrates:
  - call count verification
  - call order verification
  - exact argument matching with mock.call
  - partial matching with ANY
  - monkeypatch vs unittest.mock.patch side-by-side

Run with:
    pytest -v test_email_sending.py
"""

import pytest
from unittest.mock import Mock, call, patch, ANY

from class_a import ClassA


# ─────────────────────────────────────────────────────────────────────────────
# 1. Verify call count
# ─────────────────────────────────────────────────────────────────────────────

def test_email_sent_twice(monkeypatch):
    """Guard against accidental duplicate or missing sends."""
    email_mock = Mock(name="send_email_mock")
    monkeypatch.setattr("class_a.send_email", email_mock)

    ClassA().process()

    assert email_mock.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. Verify call order and exact arguments using assert_has_calls
# ─────────────────────────────────────────────────────────────────────────────

def test_email_order_and_exact_args(monkeypatch):
    """
    assert_has_calls validates both the arguments and the sequence.
    any_order=False means order must match exactly.
    """
    email_mock = Mock(name="send_email_mock")
    monkeypatch.setattr("class_a.send_email", email_mock)

    ClassA().process()

    expected_calls = [
        call(
            to=["user1@test.com"],
            cc=[],
            sender="noreply@test.com",
            subject="Report 1",
            body="First report",
        ),
        call(
            to=["user2@test.com"],
            cc=["manager@test.com"],
            sender="noreply@test.com",
            subject="Report 2",
            body="Second report",
        ),
    ]
    email_mock.assert_has_calls(expected_calls, any_order=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify via call_args_list (manual inspection)
# ─────────────────────────────────────────────────────────────────────────────

def test_email_call_args_list(monkeypatch):
    """
    Inspect each call individually — useful when only certain fields matter.
    """
    email_mock = Mock()
    monkeypatch.setattr("class_a.send_email", email_mock)

    ClassA().process()

    first = email_mock.call_args_list[0]
    assert first.kwargs["subject"] == "Report 1"
    assert first.kwargs["to"] == ["user1@test.com"]
    assert first.kwargs["cc"] == []

    second = email_mock.call_args_list[1]
    assert second.kwargs["subject"] == "Report 2"
    assert second.kwargs["cc"] == ["manager@test.com"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Partial matching with ANY (for dynamic content like timestamps in body)
# ─────────────────────────────────────────────────────────────────────────────

def test_email_partial_match_with_any(monkeypatch):
    """
    Use ANY when some fields are dynamic (e.g. generated timestamps, UUIDs).
    Only assert on the fields you control.
    """
    email_mock = Mock()
    monkeypatch.setattr("class_a.send_email", email_mock)

    ClassA().process()

    email_mock.assert_has_calls([
        call(
            to=["user1@test.com"],
            cc=ANY,
            sender="noreply@test.com",
            subject="Report 1",
            body=ANY,
        )
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Same test using unittest.mock.patch (context manager style)
# ─────────────────────────────────────────────────────────────────────────────

def test_email_with_patch_context_manager():
    """
    Equivalent to the monkeypatch version.
    patch auto-creates the mock; monkeypatch requires you to supply it.
    """
    with patch("class_a.send_email") as email_mock:
        ClassA().process()

        expected_calls = [
            call(
                to=["user1@test.com"],
                cc=[],
                sender="noreply@test.com",
                subject="Report 1",
                body="First report",
            ),
            call(
                to=["user2@test.com"],
                cc=["manager@test.com"],
                sender="noreply@test.com",
                subject="Report 2",
                body="Second report",
            ),
        ]
        email_mock.assert_has_calls(expected_calls, any_order=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Guard against accidental double-send
# ─────────────────────────────────────────────────────────────────────────────

def test_no_duplicate_email_to_user1(monkeypatch):
    """
    Ensure user1 only receives one email per process() call.
    Simple call_count guard prevents accidental duplicates.
    """
    email_mock = Mock()
    monkeypatch.setattr("class_a.send_email", email_mock)

    ClassA().process()

    user1_calls = [
        c for c in email_mock.call_args_list
        if c.kwargs.get("to") == ["user1@test.com"]
    ]
    assert len(user1_calls) == 1


if __name__ == "__main__":
    print("Run with: pytest -v test_email_sending.py")
