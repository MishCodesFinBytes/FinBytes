"""
test_py311_analyzer.py — Tests for the py311_readiness_analyzer tool.

Verifies that each detection rule fires correctly and that clean
code produces no false positives.

Run with:
    pytest -v test_py311_analyzer.py
"""

import textwrap
import tempfile
import os
import pytest

from py311_readiness_analyzer import analyze_file, analyze_path, Py311Analyzer
import ast


# ─────────────────────────────────────────────────────────────────────────────
# Helper: write a temp .py file and analyze it
# ─────────────────────────────────────────────────────────────────────────────

def issues_from_source(source: str) -> list[str]:
    """Write source to a temp file, run the analyzer, return issues."""
    source = textwrap.dedent(source)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        path = f.name
    try:
        return analyze_file(path)
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. None ordering comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_none_less_than_comparison():
    source = """
        x = None
        if y > x:
            pass
    """
    issues = issues_from_source(source)
    assert any("None comparison" in i for i in issues)


def test_detects_none_greater_than_comparison():
    source = """
        threshold = None
        if score > threshold:
            pass
    """
    issues = issues_from_source(source)
    assert any("None comparison" in i for i in issues)


def test_safe_equality_none_not_flagged():
    source = """
        x = None
        if x is None:
            pass
        if x == None:
            pass
    """
    issues = issues_from_source(source)
    none_order = [i for i in issues if "None comparison" in i]
    assert len(none_order) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Date/datetime arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_date_subtraction():
    source = """
        delta = today - deadline_date
    """
    issues = issues_from_source(source)
    assert any("date arithmetic" in i for i in issues)


def test_detects_date_addition():
    source = """
        end_date = start_date + duration
    """
    issues = issues_from_source(source)
    assert any("date arithmetic" in i for i in issues)


def test_plain_arithmetic_not_flagged():
    source = """
        total = price + tax
        diff = high - low
    """
    issues = issues_from_source(source)
    date_issues = [i for i in issues if "date arithmetic" in i]
    assert len(date_issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pandas .dt accessor
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_dt_accessor():
    source = """
        year = df["col"].dt.year
    """
    issues = issues_from_source(source)
    assert any("pandas .dt" in i for i in issues)


def test_other_attributes_not_flagged():
    source = """
        val = obj.value
        name = person.name
    """
    issues = issues_from_source(source)
    dt_issues = [i for i in issues if "pandas .dt" in i]
    assert len(dt_issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ambiguous truthiness
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_ambiguous_df_name():
    source = """
        if df:
            process(df)
    """
    issues = issues_from_source(source)
    assert any("ambiguous truthiness" in i for i in issues)


def test_detects_ambiguous_data_name():
    source = """
        x = data
    """
    issues = issues_from_source(source)
    assert any("ambiguous truthiness" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# 5. String/bytes — open/print/input
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_open_call():
    source = """
        with open("file.txt") as f:
            pass
    """
    issues = issues_from_source(source)
    assert any("string/bytes" in i and "open" in i for i in issues)


def test_detects_print_call():
    source = """
        print("hello world")
    """
    issues = issues_from_source(source)
    assert any("string/bytes" in i and "print" in i for i in issues)


def test_unrelated_function_not_flagged():
    source = """
        result = calculate(x, y)
        value = transform(data)
    """
    issues = issues_from_source(source)
    bytes_issues = [i for i in issues if "string/bytes" in i]
    assert len(bytes_issues) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. is vs == for string equality
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_is_string_comparison():
    source = """
        if status is "OK":
            pass
    """
    issues = issues_from_source(source)
    assert any("is vs ==" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Parse error handling
# ─────────────────────────────────────────────────────────────────────────────

def test_handles_syntax_error_gracefully():
    source = """
        def broken(:
            pass
    """
    issues = issues_from_source(source)
    assert any("parse error" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Directory scanning
# ─────────────────────────────────────────────────────────────────────────────

def test_scans_directory(tmp_path):
    """Verify analyze_path walks a directory and finds issues across files."""
    file1 = tmp_path / "module_a.py"
    file1.write_text("if score > threshold:\n    pass\n", encoding="utf-8")

    file2 = tmp_path / "module_b.py"
    file2.write_text("print('hello')\n", encoding="utf-8")

    issues = analyze_path(str(tmp_path))
    assert len(issues) >= 2
    filenames = [i.split(":")[0] for i in issues]
    assert any("module_a" in f for f in filenames)
    assert any("module_b" in f for f in filenames)


def test_skips_non_python_files(tmp_path):
    """Non-.py files should be silently skipped."""
    (tmp_path / "README.md").write_text("# Docs", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "clean.py").write_text("x = 1 + 1\n", encoding="utf-8")

    issues = analyze_path(str(tmp_path))
    # Only clean.py scanned — no issues expected from `x = 1 + 1`
    non_py = [i for i in issues if ".md" in i or ".json" in i]
    assert len(non_py) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. CI exit code behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_file_returns_no_issues(tmp_path):
    """A file with no flagged patterns should return an empty issues list."""
    clean = tmp_path / "clean.py"
    clean.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    issues = analyze_path(str(tmp_path))
    assert issues == []


if __name__ == "__main__":
    print("Run with: pytest -v test_py311_analyzer.py")
    print()
    print("Or run the analyzer directly:")
    print("  python py311_readiness_analyzer.py sample_code_with_issues.py")
