"""
test_boolean_catch22.py — Tests for boolean_catch22.py
Asserts every cell in the truth table from the blog post.

pytest test_boolean_catch22.py -v
"""

import pytest
import argparse

from boolean_catch22 import (
    build_store_true_correct,
    build_store_true_redundant,
    build_store_true_broken,
    build_store_false_correct,
    build_store_false_redundant,
    build_store_false_broken,
    build_boolean_optional_action,
    build_type_bool_broken,
    demonstrate_type_bool_bug,
    probe,
)


# ─────────────────────────────────────────────────────────────────────────────
# store_true
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreTrueCorrect:
    def test_absent_is_false(self):
        p = build_store_true_correct()
        assert p.parse_args([]).verbose is False

    def test_present_is_true(self):
        p = build_store_true_correct()
        assert p.parse_args(["--verbose"]).verbose is True

    def test_can_change(self):
        r = probe(build_store_true_correct(), "verbose", "--verbose")
        assert r["can_change"] is True


class TestStoreTrueRedundant:
    def test_absent_is_false(self):
        p = build_store_true_redundant()
        assert p.parse_args([]).debug is False

    def test_present_is_true(self):
        p = build_store_true_redundant()
        assert p.parse_args(["--debug"]).debug is True

    def test_can_change(self):
        r = probe(build_store_true_redundant(), "debug", "--debug")
        assert r["can_change"] is True


class TestStoreTrueBroken:
    """The Catch-22: flag exists but value never changes."""

    def test_absent_is_true(self):
        p = build_store_true_broken()
        assert p.parse_args([]).feature is True

    def test_present_is_also_true(self):
        p = build_store_true_broken()
        assert p.parse_args(["--feature"]).feature is True

    def test_cannot_change(self):
        r = probe(build_store_true_broken(), "feature", "--feature")
        assert r["can_change"] is False


# ─────────────────────────────────────────────────────────────────────────────
# store_false
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreFalseCorrect:
    def test_absent_is_true(self):
        p = build_store_false_correct()
        assert p.parse_args([]).use_cache is True

    def test_present_is_false(self):
        p = build_store_false_correct()
        assert p.parse_args(["--no-cache"]).use_cache is False

    def test_can_change(self):
        r = probe(build_store_false_correct(), "use_cache", "--no-cache")
        assert r["can_change"] is True


class TestStoreFalseRedundant:
    def test_absent_is_true(self):
        p = build_store_false_redundant()
        assert p.parse_args([]).use_auth is True

    def test_present_is_false(self):
        p = build_store_false_redundant()
        assert p.parse_args(["--no-auth"]).use_auth is False

    def test_can_change(self):
        r = probe(build_store_false_redundant(), "use_auth", "--no-auth")
        assert r["can_change"] is True


class TestStoreFalseBroken:
    """The inverted Catch-22: flag exists but value is always False."""

    def test_absent_is_false(self):
        p = build_store_false_broken()
        assert p.parse_args([]).logging is False

    def test_present_is_also_false(self):
        p = build_store_false_broken()
        assert p.parse_args(["--no-logging"]).logging is False

    def test_cannot_change(self):
        r = probe(build_store_false_broken(), "logging", "--no-logging")
        assert r["can_change"] is False


# ─────────────────────────────────────────────────────────────────────────────
# BooleanOptionalAction — symmetric control
# ─────────────────────────────────────────────────────────────────────────────

class TestBooleanOptionalAction:
    def test_enable_flag(self):
        p = build_boolean_optional_action()
        assert p.parse_args(["--feature"]).feature is True

    def test_disable_flag(self):
        p = build_boolean_optional_action()
        assert p.parse_args(["--no-feature"]).feature is False

    def test_default_is_true(self):
        p = build_boolean_optional_action()
        assert p.parse_args([]).feature is True

    def test_both_directions_work(self):
        p = build_boolean_optional_action()
        assert p.parse_args(["--feature"]).feature != p.parse_args(["--no-feature"]).feature


# ─────────────────────────────────────────────────────────────────────────────
# type=bool — the impossible boolean
# ─────────────────────────────────────────────────────────────────────────────

class TestTypeBoolBug:
    def test_false_string_evaluates_true(self):
        """The core bug: bool("false") is True."""
        p = build_type_bool_broken()
        assert p.parse_args(["--enabled", "false"]).enabled is True

    def test_zero_string_evaluates_true(self):
        """bool("0") is also True."""
        p = build_type_bool_broken()
        assert p.parse_args(["--enabled", "0"]).enabled is True

    def test_demonstrate_type_bool_bug_returns_true_for_both(self):
        results = demonstrate_type_bool_bug()
        assert results["--enabled false"] is True
        assert results["--enabled 0"] is True


# ─────────────────────────────────────────────────────────────────────────────
# probe helper
# ─────────────────────────────────────────────────────────────────────────────

def test_probe_reports_correct_status():
    r = probe(build_store_true_correct(), "verbose", "--verbose")
    assert r["absent"] is False
    assert r["present"] is True
    assert r["can_change"] is True
    assert "CORRECT" in r["status"]


def test_probe_reports_broken_status():
    r = probe(build_store_true_broken(), "feature", "--feature")
    assert r["absent"] == r["present"]
    assert r["can_change"] is False
    assert "CATCH-22" in r["status"]
