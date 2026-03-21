"""
test_check_parser_universal.py — Tests for check_parser_universal.py

pytest test_check_parser_universal.py -v
"""

import argparse
import pytest

from check_parser_universal import (
    check_parser,
    load_parser_from_path,
    main,
    Issue,
)
from sample_parsers import broken_parser, clean_parser, subcommand_parser_with_issues


# ─────────────────────────────────────────────────────────────────────────────
# check_parser — broken_parser
# ─────────────────────────────────────────────────────────────────────────────

class TestBrokenParser:
    def setup_method(self):
        self.issues = check_parser(broken_parser())
        self.categories = [i.category for i in self.issues]

    def test_finds_boolean_catch22(self):
        assert "Catch-22" in self.categories

    def test_finds_type_bool(self):
        assert "TypeBool" in self.categories

    def test_finds_required_with_default(self):
        assert "RequiredWithDefault" in self.categories

    def test_verbose_flag_flagged(self):
        flags = [i.flag for i in self.issues]
        assert "--verbose" in flags

    def test_force_flag_flagged(self):
        flags = [i.flag for i in self.issues]
        assert "--force" in flags

    def test_enabled_flag_flagged(self):
        flags = [i.flag for i in self.issues]
        assert "--enabled" in flags

    def test_mode_flag_flagged(self):
        flags = [i.flag for i in self.issues]
        assert "--mode" in flags

    def test_returns_multiple_issues(self):
        assert len(self.issues) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# check_parser — clean_parser
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanParser:
    def test_no_issues_on_clean_parser(self):
        issues = check_parser(clean_parser())
        # --output is required but has no default — should pass
        # --no-cache is store_false with no explicit default — should pass
        assert issues == []


# ─────────────────────────────────────────────────────────────────────────────
# check_parser — individual issue types
# ─────────────────────────────────────────────────────────────────────────────

def test_detects_store_true_catch22():
    p = argparse.ArgumentParser()
    p.add_argument("--flag", action="store_true", default=True)
    issues = check_parser(p)
    assert any(i.category == "Catch-22" and i.flag == "--flag" for i in issues)


def test_does_not_flag_correct_store_true():
    p = argparse.ArgumentParser()
    p.add_argument("--flag", action="store_true")
    issues = check_parser(p)
    assert not any(i.category == "Catch-22" for i in issues)


def test_detects_store_false_catch22():
    p = argparse.ArgumentParser()
    p.add_argument("--no-log", action="store_false", dest="logging", default=False)
    issues = check_parser(p)
    assert any(i.category == "Catch-22" for i in issues)


def test_does_not_flag_correct_store_false():
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_false", dest="use_cache")
    issues = check_parser(p)
    assert not any(i.category == "Catch-22" for i in issues)


def test_detects_type_bool():
    p = argparse.ArgumentParser()
    p.add_argument("--enabled", type=bool)
    issues = check_parser(p)
    assert any(i.category == "TypeBool" for i in issues)


def test_detects_required_with_default():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, default="dev")
    issues = check_parser(p)
    assert any(i.category == "RequiredWithDefault" for i in issues)


def test_detects_single_choice():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["prod"], default="prod")
    issues = check_parser(p)
    assert any(i.category == "SingleChoice" for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# Subparser recursion
# ─────────────────────────────────────────────────────────────────────────────

def test_recurses_into_subparsers():
    issues = check_parser(subcommand_parser_with_issues())
    assert any(i.flag == "--debug" for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# load_parser_from_path
# ─────────────────────────────────────────────────────────────────────────────

def test_load_valid_path():
    parser = load_parser_from_path("sample_parsers.clean_parser")
    assert isinstance(parser, argparse.ArgumentParser)


def test_load_broken_parser():
    parser = load_parser_from_path("sample_parsers.broken_parser")
    assert isinstance(parser, argparse.ArgumentParser)


def test_load_missing_module_raises():
    with pytest.raises(ImportError):
        load_parser_from_path("nonexistent.module.build_parser")


def test_load_missing_function_raises():
    with pytest.raises(AttributeError):
        load_parser_from_path("sample_parsers.nonexistent_function")


# ─────────────────────────────────────────────────────────────────────────────
# main() CLI function
# ─────────────────────────────────────────────────────────────────────────────

def test_main_returns_1_for_broken_parser():
    exit_code = main(["sample_parsers.broken_parser"])
    assert exit_code == 1


def test_main_returns_0_for_clean_parser():
    exit_code = main(["sample_parsers.clean_parser"])
    assert exit_code == 0


def test_main_returns_2_for_missing_arg():
    exit_code = main([])
    assert exit_code == 2


def test_main_returns_2_for_bad_module():
    exit_code = main(["nonexistent.module.fn"])
    assert exit_code == 2


# ─────────────────────────────────────────────────────────────────────────────
# Issue dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_issue_str_format():
    issue = Issue(category="Catch-22", flag="--verbose", detail="always True")
    assert "[Catch-22]" in str(issue)
    assert "--verbose" in str(issue)
    assert "always True" in str(issue)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full scan matches blog post output
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_parser_issue_flags_match_expected():
    """Mirrors the expected output from the blog post."""
    issues = check_parser(broken_parser())
    flags = {i.flag for i in issues}
    assert "--verbose" in flags
    assert "--force"   in flags
    assert "--enabled" in flags
    assert "--mode"    in flags
