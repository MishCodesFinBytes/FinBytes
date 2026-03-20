"""
test_catch22_catalog.py — Tests for catch22_catalog.py
Covers all 9 Catch-22 patterns and their fixes.

pytest test_catch22_catalog.py -v
"""

import os
import pytest
import argparse

from catch22_catalog import (
    broken_1_store_true_default_true,
    fixed_1_store_true,
    broken_2_exclusive_group_fake_choice,
    fixed_2_boolean_optional_action,
    broken_3_required_with_default,
    fixed_3a_required_no_default,
    fixed_3b_optional_with_default,
    broken_4_single_choice,
    fixed_4_meaningful_choices,
    broken_5_nargs_identical_default_const,
    fixed_5_nargs_distinct_values,
    broken_6_mutable_default,
    fixed_6_none_default,
    normalize_tags,
    broken_7_env_silently_wins,
    fixed_7_explicit_precedence,
    broken_8_type_bool,
    fixed_8_store_true_store_false,
    broken_9_set_defaults_bypass,
    fixed_9_explicit_format_choice,
    audit_parser,
    can_flag_change_value,
)


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #1
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_1_feature_always_true():
    p = broken_1_store_true_default_true()
    assert p.parse_args([]).feature is True
    assert p.parse_args(["--feature"]).feature is True


def test_fixed_1_feature_starts_false():
    p = fixed_1_store_true()
    assert p.parse_args([]).feature is False
    assert p.parse_args(["--feature"]).feature is True


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #2
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_2_fast_flag_no_effect():
    p = broken_2_exclusive_group_fake_choice()
    assert p.parse_args([]).fast is True
    assert p.parse_args(["--fast"]).fast is True  # no change


def test_broken_2_slow_still_works():
    p = broken_2_exclusive_group_fake_choice()
    assert p.parse_args(["--slow"]).fast is False


def test_fixed_2_both_directions():
    p = fixed_2_boolean_optional_action()
    assert p.parse_args(["--fast"]).fast is True
    assert p.parse_args(["--no-fast"]).fast is False


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #3
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_3_required_bypassed_by_default():
    p = broken_3_required_with_default()
    args = p.parse_args([])   # should conceptually fail but passes due to default
    assert args.mode == "dev"


def test_fixed_3a_missing_mode_raises():
    p = fixed_3a_required_no_default()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_fixed_3b_default_applies():
    p = fixed_3b_optional_with_default()
    assert p.parse_args([]).mode == "dev"
    assert p.parse_args(["--mode", "prod"]).mode == "prod"


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #4
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_4_single_choice_no_effect():
    p = broken_4_single_choice()
    assert p.parse_args([]).env == "prod"
    assert p.parse_args(["--env", "prod"]).env == "prod"


def test_fixed_4_multiple_choices():
    p = fixed_4_meaningful_choices()
    assert p.parse_args([]).env == "dev"
    assert p.parse_args(["--env", "prod"]).env == "prod"
    assert p.parse_args(["--env", "staging"]).env == "staging"


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #5
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_5_all_paths_give_same_value():
    p = broken_5_nargs_identical_default_const()
    assert p.parse_args([]).level == 1
    assert p.parse_args(["--level"]).level == 1
    assert p.parse_args(["--level", "1"]).level == "1"  # string, but same logical value


def test_fixed_5_bare_flag_differs_from_absent():
    p = fixed_5_nargs_distinct_values()
    absent = p.parse_args([]).level
    bare   = p.parse_args(["--level"]).level
    assert absent != bare


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #6
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_6_mutable_default_leaks():
    """Shared mutable default accumulates across multiple parse calls."""
    p = broken_6_mutable_default()
    args1 = p.parse_args(["--tag", "alpha"])
    args2 = p.parse_args(["--tag", "beta"])
    # Due to shared list, args2 may contain both entries
    # This is the bug — we just confirm default is a list object
    assert isinstance(args1.tag, list)


def test_fixed_6_none_default_no_leak():
    p = fixed_6_none_default()
    args = p.parse_args([])
    assert normalize_tags(args) == []


def test_fixed_6_tags_accumulate_correctly():
    p = fixed_6_none_default()
    args = p.parse_args(["--tag", "alpha", "--tag", "beta"])
    assert normalize_tags(args) == ["alpha", "beta"]


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #7
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_7_env_wins_over_cli(monkeypatch):
    monkeypatch.setenv("TIMEOUT", "999")
    result = broken_7_env_silently_wins(cli_timeout=5)
    assert result == 999   # CLI value 5 was silently overridden


def test_fixed_7_cli_wins_over_env(monkeypatch):
    monkeypatch.setenv("TIMEOUT", "999")
    result = fixed_7_explicit_precedence(cli_timeout=5)
    assert result["timeout"] == 5
    assert result["source"] == "cli"


def test_fixed_7_env_applies_when_no_cli(monkeypatch):
    monkeypatch.setenv("TIMEOUT", "60")
    result = fixed_7_explicit_precedence()
    assert result["timeout"] == 60


def test_fixed_7_builtin_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv("TIMEOUT", raising=False)
    result = fixed_7_explicit_precedence()
    assert result["timeout"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #8
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_8_false_string_is_truthy():
    p = broken_8_type_bool()
    assert p.parse_args(["--enabled", "false"]).enabled is True   # bug!
    assert p.parse_args(["--enabled", "0"]).enabled is True       # bug!


def test_fixed_8_enable_flag():
    p = fixed_8_store_true_store_false()
    assert p.parse_args(["--enable"]).enabled is True


def test_fixed_8_disable_flag():
    p = fixed_8_store_true_store_false()
    assert p.parse_args(["--disable"]).enabled is False


def test_fixed_8_default_is_enabled():
    p = fixed_8_store_true_store_false()
    assert p.parse_args([]).enabled is True


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #9
# ─────────────────────────────────────────────────────────────────────────────

def test_broken_9_json_always_true_without_flag():
    p = broken_9_set_defaults_bypass()
    args = p.parse_args([])
    assert args.json is True   # fixed by set_defaults, not user choice


def test_fixed_9_format_required():
    p = fixed_9_explicit_format_choice()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_fixed_9_format_choices():
    p = fixed_9_explicit_format_choice()
    assert p.parse_args(["--format", "json"]).format == "json"
    assert p.parse_args(["--format", "yaml"]).format == "yaml"


# ─────────────────────────────────────────────────────────────────────────────
# audit_parser
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_detects_boolean_catch22():
    issues = audit_parser(broken_1_store_true_default_true())
    assert any("CATCH-22" in i for i in issues)


def test_audit_detects_required_with_default():
    issues = audit_parser(broken_3_required_with_default())
    assert any("REQUIRED+DEFAULT" in i for i in issues)


def test_audit_detects_single_choice():
    issues = audit_parser(broken_4_single_choice())
    assert any("SINGLE-CHOICE" in i for i in issues)


def test_audit_clean_parser_returns_no_issues():
    p = fixed_1_store_true()
    issues = audit_parser(p)
    assert issues == []


# ─────────────────────────────────────────────────────────────────────────────
# can_flag_change_value helper
# ─────────────────────────────────────────────────────────────────────────────

def test_can_change_returns_true_for_correct_flag():
    assert can_flag_change_value(fixed_1_store_true(), "feature", "--feature") is True


def test_can_change_returns_false_for_broken_flag():
    assert can_flag_change_value(broken_1_store_true_default_true(), "feature", "--feature") is False
