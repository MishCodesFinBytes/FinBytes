"""
test_external_side_effects.py — Runnable demo for:
"Testing External Side Effects: Files, Emails, and APIs — A Unified Strategy"

Covers: patching file writes, email sends, and HTTP API calls.
All tests run without touching disk, network, or a mail server.

Run with:
    pytest -v test_external_side_effects.py
"""

import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import Mock, call, patch

from class_a import ClassA
from api_client import APIClient


# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE OUTPUT — patch DataFrame.to_excel (no disk writes)
# ─────────────────────────────────────────────────────────────────────────────

class ReportWriter:
    def save_report(self, df: pd.DataFrame, file_path: str):
        df.to_excel(file_path, index=False)


def test_save_report_patches_to_excel(monkeypatch):
    """Verify to_excel is called with the correct path and kwargs."""
    mock_to_excel = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_excel", mock_to_excel)

    df = pd.DataFrame({"a": [1, 2]})
    writer = ReportWriter()
    writer.save_report(df, "output.xlsx")

    mock_to_excel.assert_called_once_with("output.xlsx", index=False)


def test_save_report_content_via_buffer(monkeypatch):
    """Redirect to BytesIO and verify the actual Excel content."""
    buffer = BytesIO()
    original_to_excel = pd.DataFrame.to_excel

    def fake_to_excel(self, file_path, index=False):
        original_to_excel(self, buffer, index=index)

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    writer = ReportWriter()
    writer.save_report(df, "ignored.xlsx")

    buffer.seek(0)
    result_df = pd.read_excel(buffer)
    assert result_df.equals(df)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMAIL SENDING — verify call count, order, and arguments
# ─────────────────────────────────────────────────────────────────────────────

def test_email_calls_count_and_order(monkeypatch):
    """Verify exactly 2 emails are sent in the correct order."""
    email_mock = Mock()
    monkeypatch.setattr("class_a.send_email", email_mock)

    obj = ClassA()
    obj.process()

    assert email_mock.call_count == 2

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


def test_email_individual_call_args(monkeypatch):
    """Inspect individual call arguments directly."""
    email_mock = Mock()
    monkeypatch.setattr("class_a.send_email", email_mock)

    ClassA().process()

    first_call = email_mock.call_args_list[0]
    assert first_call.kwargs["subject"] == "Report 1"
    assert first_call.kwargs["to"] == ["user1@test.com"]

    second_call = email_mock.call_args_list[1]
    assert second_call.kwargs["cc"] == ["manager@test.com"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTTP API CALLS — patch requests.get, no real network
# ─────────────────────────────────────────────────────────────────────────────

def test_fetch_data_returns_mocked_response(monkeypatch):
    """Patch requests.get and verify the client returns parsed JSON."""
    mock_get = Mock()
    mock_get.return_value.json.return_value = {"data": 42}
    monkeypatch.setattr("requests.get", mock_get)

    client = APIClient()
    result = client.fetch_data("http://example.com/api")

    assert result == {"data": 42}
    mock_get.assert_called_once_with("http://example.com/api")


def test_fetch_data_uses_correct_url(monkeypatch):
    """Ensure the client forwards the URL unchanged."""
    mock_get = Mock()
    mock_get.return_value.json.return_value = {}
    monkeypatch.setattr("requests.get", mock_get)

    client = APIClient()
    client.fetch_data("https://api.finbytes.com/prices")

    mock_get.assert_called_once_with("https://api.finbytes.com/prices")


if __name__ == "__main__":
    print("Run with: pytest -v test_external_side_effects.py")
