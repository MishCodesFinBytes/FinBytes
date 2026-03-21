"""
test_migration.py — Runnable demo for:
"Migrating from Python 3.6 to 3.11: A Practical Enterprise Guide"

Each test demonstrates one item from the migration checklist.
Tests are written in the 3.11-safe style throughout.

Run with:
    pytest -v test_migration.py
"""

import math
import time
import pytest
from datetime import date, datetime

from fixed_311 import (
    days_until,
    get_unique_tags,
    is_above_threshold,
    parse_amount,
    compound_rate,
    rates_are_equal,
    read_file,
    write_file,
    check_status,
    get_config_keys,
    wait_for_condition,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Date/datetime arithmetic
# ─────────────────────────────────────────────────────────────────────────────

class TestDateArithmetic:
    def test_days_until_future_date_is_positive(self):
        """Deadline in the future → positive day count."""
        far_future = date(2099, 12, 31)
        assert days_until(far_future) > 0

    def test_days_until_past_date_is_negative(self):
        """Deadline in the past → negative day count."""
        past = date(2000, 1, 1)
        assert days_until(past) < 0

    def test_no_type_error_mixing_date_and_datetime(self):
        """The fixed version converts explicitly — no TypeError raised."""
        try:
            days_until(date.today())
        except TypeError:
            pytest.fail("days_until() raised TypeError — date/datetime mixing not handled")

    def test_pandas_datetime_conversion(self):
        """Demonstrate the pd.to_datetime fix pattern without a real DataFrame."""
        import pandas as pd
        s = pd.Series(["2024-01-01", "2024-06-15"])
        converted = pd.to_datetime(s)
        # .dt accessor is now safe
        years = converted.dt.year.tolist()
        assert years == [2024, 2024]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Set and dict ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestOrdering:
    def test_unique_tags_sorted(self):
        """get_unique_tags returns a sorted, deduplicated list."""
        result = get_unique_tags(["banana", "apple", "cherry", "apple"])
        assert result == ["apple", "banana", "cherry"]

    def test_set_comparison_use_sorted(self):
        """Never compare raw sets to lists — sort first."""
        raw = {"c", "a", "b"}
        assert sorted(raw) == ["a", "b", "c"]

    def test_dict_key_order_deterministic(self):
        """get_config_keys sorts keys for deterministic comparisons."""
        config = {"z_key": 1, "a_key": 2, "m_key": 3}
        assert get_config_keys(config) == ["a_key", "m_key", "z_key"]

    def test_set_equality_correct_approach(self):
        """Compare sets to sets for membership, not order."""
        result_set = set(get_unique_tags(["b", "a", "c"]))
        expected = {"a", "b", "c"}
        assert result_set == expected


# ─────────────────────────────────────────────────────────────────────────────
# 3. None comparisons and truthiness
# ─────────────────────────────────────────────────────────────────────────────

class TestNoneAndTruthiness:
    def test_threshold_none_returns_false(self):
        """None threshold is handled gracefully — no TypeError."""
        assert is_above_threshold(10, threshold=None) is False

    def test_threshold_set_works_correctly(self):
        assert is_above_threshold(10, threshold=5) is True
        assert is_above_threshold(3, threshold=5) is False

    def test_none_is_not_ordering_comparison(self):
        """Explicit is None check — safe in all versions."""
        x = None
        assert x is None
        assert not (x is not None)

    def test_dataframe_empty_check(self):
        """Use .empty instead of bare truthiness on DataFrames."""
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert not df.empty

        empty_df = pd.DataFrame()
        assert empty_df.empty


# ─────────────────────────────────────────────────────────────────────────────
# 4. Exception handling — type, not message
# ─────────────────────────────────────────────────────────────────────────────

class TestExceptionHandling:
    def test_parse_amount_raises_value_error(self):
        """Assert on exception type only — message text changes between versions."""
        with pytest.raises(ValueError):
            parse_amount("not_a_number")

    def test_parse_amount_valid_input(self):
        assert parse_amount("42") == 42

    def test_exception_type_not_message(self):
        """Demonstrate the safe pattern: match=r'' is optional and fragile."""
        with pytest.raises(ZeroDivisionError):
            _ = 1 / 0

    def test_exception_match_on_type_only(self):
        """If you must match text, use a very minimal pattern."""
        with pytest.raises(ValueError, match="invalid"):
            int("abc")
        # Note: even this can break — prefer type-only assertions


# ─────────────────────────────────────────────────────────────────────────────
# 5. Float and decimal precision
# ─────────────────────────────────────────────────────────────────────────────

class TestFloatPrecision:
    def test_compound_rate_with_tolerance(self):
        """Use math.isclose instead of exact equality for floats."""
        result = compound_rate(1000, 0.1, 3)
        assert math.isclose(result, 1331.0, rel_tol=1e-9)

    def test_rates_are_equal_helper(self):
        assert rates_are_equal(0.1 + 0.2, 0.3) is True

    def test_float_equality_exact_fails(self):
        """Show why exact equality is dangerous."""
        # 0.1 + 0.2 is NOT exactly 0.3 in IEEE 754
        assert (0.1 + 0.2) != 0.3  # this is expected!

    def test_float_equality_safe(self):
        assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. String and bytes — explicit encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestStringBytes:
    def test_read_write_file_utf8(self, tmp_path):
        """Explicit encoding= prevents platform-dependent surprises."""
        path = str(tmp_path / "test.txt")
        write_file(path, "Hello, FinBytes 🐍")
        content = read_file(path)
        assert content == "Hello, FinBytes 🐍"

    def test_bytes_encoding_explicit(self):
        """Always encode/decode explicitly when mixing str and bytes."""
        text = "hello"
        encoded = text.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == text

    def test_no_implicit_bytes_concat(self):
        """Demonstrates that bytes + str raises TypeError — catch it early."""
        with pytest.raises(TypeError):
            _ = b"prefix" + "suffix"   # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 7. is vs == for value equality
# ─────────────────────────────────────────────────────────────────────────────

class TestIsVsEquals:
    def test_status_ok_with_equals(self):
        """== checks value — correct and reliable in all Python versions."""
        assert check_status("OK") is True

    def test_status_not_ok(self):
        assert check_status("FAIL") is False

    def test_is_identity_not_equality(self):
        """Demonstrate that `is` checks identity, not value."""
        a = "hello world"      # long string — unlikely to be interned
        b = "hello " + "world"
        # == is always correct
        assert a == b
        # `is` is unreliable — don't use for strings
        # (may be True or False depending on interpreter optimisation)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Timing — poll instead of sleep
# ─────────────────────────────────────────────────────────────────────────────

class TestTiming:
    def test_wait_for_condition_succeeds(self):
        """Condition becomes true after first poll — should return True."""
        calls = {"n": 0}

        def condition():
            calls["n"] += 1
            return calls["n"] >= 2

        result = wait_for_condition(condition, timeout=2.0, interval=0.01)
        assert result is True

    def test_wait_for_condition_times_out(self):
        """Condition never becomes true — should return False after timeout."""
        result = wait_for_condition(lambda: False, timeout=0.1, interval=0.05)
        assert result is False

    def test_no_hardcoded_sleep(self):
        """
        Demonstrate why sleep-based tests are fragile on 3.11 (faster interpreter).
        Use wait_for_condition with a real condition instead.
        """
        ready = {"flag": False}

        def set_flag():
            ready["flag"] = True

        set_flag()
        result = wait_for_condition(lambda: ready["flag"], timeout=1.0)
        assert result is True


# ─────────────────────────────────────────────────────────────────────────────
# 9. Warnings — run with -W error to surface deprecations
# ─────────────────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_no_deprecation_warnings_raised(self, recwarn):
        """
        Verify that fixed_311 functions don't emit deprecation warnings.
        In CI, run pytest -W error to treat warnings as failures.
        """
        get_unique_tags(["a", "b"])
        is_above_threshold(5, 3)
        check_status("OK")

        deprecation_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0


if __name__ == "__main__":
    print("Run with: pytest -v test_migration.py")
    print()
    print("For strict CI mode (warnings as errors):")
    print("  pytest -W error --strict-markers test_migration.py")
