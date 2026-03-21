"""
test_file_output.py — Runnable demo for:
"Testing File Output in Python Without Touching the Filesystem"

Five strategies demonstrated:
  1. Patch to_excel — verify call and args only
  2. Redirect to BytesIO — verify Excel content in memory
  3. Patch your own abstraction layer
  4. unittest.mock.patch equivalent
  5. tmp_path for real integration-style tests

Run with:
    pytest -v test_file_output.py
"""

import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import Mock, patch

from report_writer import ReportWriter


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Patch to_excel: verify call, path, and kwargs only
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy1_patch_call_verification(monkeypatch):
    """
    Pure unit test: did we call to_excel with the right arguments?
    No file is created. Fast, deterministic.
    """
    mock_to_excel = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_excel", mock_to_excel)

    df = pd.DataFrame({"col": [10, 20, 30]})
    writer = ReportWriter()
    writer.save_report(df, "my_report.xlsx")

    mock_to_excel.assert_called_once_with("my_report.xlsx", index=False)


def test_strategy1_index_false_enforced(monkeypatch):
    """Verify index=False is always passed (common requirement)."""
    mock_to_excel = Mock()
    monkeypatch.setattr(pd.DataFrame, "to_excel", mock_to_excel)

    writer = ReportWriter()
    writer.save_report(pd.DataFrame({"a": [1]}), "out.xlsx")

    _, kwargs = mock_to_excel.call_args
    assert kwargs.get("index") is False


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Redirect to BytesIO: validate Excel content in memory
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy2_excel_content_in_buffer(monkeypatch):
    """
    High-confidence test: redirects the write to an in-memory buffer,
    then reads it back to verify the data survived the round-trip.
    """
    buffer = BytesIO()
    original_to_excel = pd.DataFrame.to_excel

    def fake_to_excel(self, file_path, index=False):
        original_to_excel(self, buffer, index=index)

    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [95, 87]})
    writer = ReportWriter()
    writer.save_report(df, "ignored.xlsx")

    buffer.seek(0)
    result = pd.read_excel(buffer)
    assert result.equals(df)


def test_strategy2_column_names_preserved(monkeypatch):
    """Verify column names survive the Excel serialization."""
    buffer = BytesIO()
    original = pd.DataFrame.to_excel

    def capture(self, path, index=False):
        original(self, buffer, index=index)

    monkeypatch.setattr(pd.DataFrame, "to_excel", capture)

    df = pd.DataFrame({"amount": [100.5], "currency": ["GBP"]})
    ReportWriter().save_report(df, "x.xlsx")

    buffer.seek(0)
    result = pd.read_excel(buffer)
    assert list(result.columns) == ["amount", "currency"]


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Patch your own abstraction layer (cleaner architecture)
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy3_patch_wrapper_method(monkeypatch):
    """
    Patch the thin _write_excel wrapper instead of pandas internals.
    More stable — insulates tests from pandas API changes.
    """
    mock_write = Mock()
    monkeypatch.setattr(ReportWriter, "_write_excel", mock_write)

    df = pd.DataFrame({"x": [1, 2]})
    writer = ReportWriter()
    writer.save_report_via_wrapper(df, "report.xlsx")

    mock_write.assert_called_once()
    call_args = mock_write.call_args
    assert call_args.args[1] == "report.xlsx"  # path argument


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4 — unittest.mock.patch equivalent (context manager style)
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy4_patch_object_context_manager():
    """
    Equivalent to Strategy 1 but using patch.object context manager.
    Functionally identical — choose based on team style preference.
    """
    with patch.object(pd.DataFrame, "to_excel") as mock_to_excel:
        df = pd.DataFrame({"a": [1, 2]})
        writer = ReportWriter()
        writer.save_report(df, "file.xlsx")

        mock_to_excel.assert_called_once_with("file.xlsx", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5 — tmp_path: real file, temporary directory, no residue
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy5_integration_with_tmp_path(tmp_path):
    """
    Integration-level test: writes a real Excel file to a pytest-managed
    temp directory. File is cleaned up automatically after the test session.
    Use for end-to-end validation, not for every unit test.
    """
    df = pd.DataFrame({"product": ["Widget", "Gadget"], "units": [100, 250]})
    writer = ReportWriter()

    file_path = tmp_path / "sales_report.xlsx"
    writer.save_report(df, file_path)

    assert file_path.exists()

    result = pd.read_excel(file_path)
    assert result.equals(df)


if __name__ == "__main__":
    print("Run with: pytest -v test_file_output.py")
