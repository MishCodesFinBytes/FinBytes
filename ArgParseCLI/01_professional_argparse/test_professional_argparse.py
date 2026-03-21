"""
test_professional_argparse.py — Tests for professional_argparse.py

pytest test_professional_argparse.py -v
"""

import pytest
import argparse

from professional_argparse import (
    build_parser,
    dispatch,
    positive_int,
    handle_process,
    handle_batch,
    handle_report,
)


@pytest.fixture
def parser():
    return build_parser()


def parse(parser, args: list[str]) -> argparse.Namespace:
    return parser.parse_args(args)


# ─────────────────────────────────────────────────────────────────────────────
# process sub-command
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessCommand:
    def test_minimal_args(self, parser):
        args = parse(parser, ["process", "data.csv"])
        assert args.input_file == "data.csv"
        assert args.env == "dev"
        assert args.workers == 1
        assert args.verbose is False
        assert args.dry_run is False

    def test_env_choices(self, parser):
        for env in ["dev", "staging", "prod"]:
            args = parse(parser, ["process", "f.csv", "--env", env])
            assert args.env == env

    def test_invalid_env_raises(self, parser):
        with pytest.raises(SystemExit):
            parse(parser, ["process", "f.csv", "--env", "local"])

    def test_verbose_flag(self, parser):
        args = parse(parser, ["process", "f.csv", "--verbose"])
        assert args.verbose is True

    def test_dry_run_flag(self, parser):
        args = parse(parser, ["process", "f.csv", "--dry-run"])
        assert args.dry_run is True

    def test_workers_positive_int(self, parser):
        args = parse(parser, ["process", "f.csv", "--workers", "4"])
        assert args.workers == 4

    def test_workers_zero_raises(self, parser):
        with pytest.raises(SystemExit):
            parse(parser, ["process", "f.csv", "--workers", "0"])

    def test_workers_negative_raises(self, parser):
        with pytest.raises(SystemExit):
            parse(parser, ["process", "f.csv", "--workers", "-1"])

    def test_tags_nargs(self, parser):
        args = parse(parser, ["process", "f.csv", "--tags", "alpha", "beta"])
        assert args.tags == ["alpha", "beta"]

    def test_tags_default_none(self, parser):
        args = parse(parser, ["process", "f.csv"])
        assert args.tags is None

    def test_handle_process_result(self, parser):
        args = parse(parser, ["process", "data.csv", "--env", "prod", "--workers", "2"])
        result = handle_process(args)
        assert result["command"] == "process"
        assert result["env"] == "prod"
        assert result["workers"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# batch sub-command
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchCommand:
    def test_required_dates(self, parser):
        args = parse(parser, ["batch", "--start-date", "2024-01-01", "--end-date", "2024-03-31"])
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-03-31"

    def test_missing_start_date_raises(self, parser):
        with pytest.raises(SystemExit):
            parse(parser, ["batch", "--end-date", "2024-03-31"])

    def test_no_cache_flag(self, parser):
        args = parse(parser, ["batch", "--start-date", "2024-01-01", "--end-date", "2024-01-31", "--no-cache"])
        assert args.use_cache is False

    def test_cache_enabled_by_default(self, parser):
        args = parse(parser, ["batch", "--start-date", "2024-01-01", "--end-date", "2024-01-31"])
        assert args.use_cache is True

    def test_handle_batch_result(self, parser):
        args = parse(parser, ["batch", "--start-date", "2024-01-01", "--end-date", "2024-12-31", "--dry-run"])
        result = handle_batch(args)
        assert result["dry_run"] is True
        assert result["use_cache"] is True


# ─────────────────────────────────────────────────────────────────────────────
# report sub-command — mutually exclusive format flags
# ─────────────────────────────────────────────────────────────────────────────

class TestReportCommand:
    def test_csv_format(self, parser):
        args = parse(parser, ["report", "--csv"])
        assert args.csv is True

    def test_json_format(self, parser):
        args = parse(parser, ["report", "--json"])
        assert args.json is True

    def test_html_format(self, parser):
        args = parse(parser, ["report", "--html"])
        assert args.html is True

    def test_no_format_raises(self, parser):
        with pytest.raises(SystemExit):
            parse(parser, ["report"])

    def test_two_formats_raises(self, parser):
        with pytest.raises(SystemExit):
            parse(parser, ["report", "--csv", "--json"])

    def test_retries_default(self, parser):
        args = parse(parser, ["report", "--csv"])
        assert args.retries == 3

    def test_retries_custom(self, parser):
        args = parse(parser, ["report", "--json", "--retries", "5"])
        assert args.retries == 5

    def test_handle_report_csv(self, parser):
        args = parse(parser, ["report", "--csv"])
        result = handle_report(args)
        assert result["format"] == "csv"

    def test_handle_report_json(self, parser):
        args = parse(parser, ["report", "--json"])
        result = handle_report(args)
        assert result["format"] == "json"


# ─────────────────────────────────────────────────────────────────────────────
# Custom type validator
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveInt:
    def test_valid_values(self):
        assert positive_int("1") == 1
        assert positive_int("100") == 100

    def test_zero_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_negative_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("-5")

    def test_non_integer_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_int("abc")


# ─────────────────────────────────────────────────────────────────────────────
# dispatch
# ─────────────────────────────────────────────────────────────────────────────

def test_dispatch_process(parser):
    args = parse(parser, ["process", "data.csv"])
    result = dispatch(args)
    assert result["command"] == "process"


def test_dispatch_batch(parser):
    args = parse(parser, ["batch", "--start-date", "2024-01-01", "--end-date", "2024-12-31"])
    result = dispatch(args)
    assert result["command"] == "batch"


def test_dispatch_report(parser):
    args = parse(parser, ["report", "--html"])
    result = dispatch(args)
    assert result["command"] == "report"
    assert result["format"] == "html"


# ─────────────────────────────────────────────────────────────────────────────
# No-arg / missing command raises
# ─────────────────────────────────────────────────────────────────────────────

def test_no_command_raises(parser):
    with pytest.raises(SystemExit):
        parse(parser, [])
