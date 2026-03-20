"""
professional_argparse.py — Runnable demo for:
"Making Python Scripts Professional: A Practical Guide to argparse"

Covers every argument type and pattern from the post:
  - positional vs optional arguments
  - boolean flags (correct: store_true, incorrect: type=bool)
  - restricted choices
  - numeric arguments with manual validation
  - nargs (multiple values)
  - mutually exclusive groups
  - subcommands
  - environment variable defaults
  - custom type validators
  - build_parser() factory pattern (testable)

Run:
    python professional_argparse.py --help
    python professional_argparse.py process data.csv --env prod --workers 4
    python professional_argparse.py batch --start-date 2024-01-01 --end-date 2024-03-31
    python professional_argparse.py report --csv

Tests:
    pytest test_professional_argparse.py -v
"""

import argparse
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Custom type validator — used with type= to enforce positive ints
# ─────────────────────────────────────────────────────────────────────────────

def positive_int(value: str) -> int:
    """Validates that a string argument is a positive integer."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer (got {ivalue})")
    return ivalue


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: process
# ─────────────────────────────────────────────────────────────────────────────

def build_process_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "process",
        help="Process an input file",
        description="Load and process a data file with configurable options.",
    )
    # Positional: what you are operating on
    p.add_argument("input_file", help="Path to the input CSV or JSON file")

    # Optional: how you operate
    p.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Deployment environment (default: dev)",
    )
    p.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="Number of parallel workers (must be > 0)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate processing without writing output",
    )
    p.add_argument(
        "--tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="One or more tags to attach to this run",
    )
    # Environment variable default
    p.add_argument(
        "--db-host",
        default=os.getenv("DB_HOST", "localhost"),
        help="Database host (default: $DB_HOST or localhost)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: batch
# ─────────────────────────────────────────────────────────────────────────────

def build_batch_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "batch",
        help="Run a batch job over a date range",
        description="Execute a batch processing job for the specified date window.",
    )
    p.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end-date",   required=True, help="End date   (YYYY-MM-DD)")
    p.add_argument("--dry-run",    action="store_true", help="Simulate without writing")
    p.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable caching for this run",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: report
# ─────────────────────────────────────────────────────────────────────────────

def build_report_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "report",
        help="Generate a report in one output format",
    )
    # Mutually exclusive: user must pick exactly one output format
    fmt = p.add_mutually_exclusive_group(required=True)
    fmt.add_argument("--csv",  action="store_true", help="Output as CSV")
    fmt.add_argument("--json", action="store_true", help="Output as JSON")
    fmt.add_argument("--html", action="store_true", help="Output as HTML")

    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of delivery retries on failure",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main parser factory — separated from execution for testability
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """
    Returns a fully configured ArgumentParser.
    Keeping this separate from __main__ enables clean unit testing.
    """
    parser = argparse.ArgumentParser(
        description="FinBytes professional CLI tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_process_parser(subparsers)
    build_batch_parser(subparsers)
    build_report_parser(subparsers)

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_process(args: argparse.Namespace) -> dict:
    return {
        "command":    "process",
        "input_file": args.input_file,
        "env":        args.env,
        "workers":    args.workers,
        "verbose":    args.verbose,
        "dry_run":    args.dry_run,
        "tags":       args.tags or [],
        "db_host":    args.db_host,
    }


def handle_batch(args: argparse.Namespace) -> dict:
    return {
        "command":    "batch",
        "start_date": args.start_date,
        "end_date":   args.end_date,
        "dry_run":    args.dry_run,
        "use_cache":  args.use_cache,
    }


def handle_report(args: argparse.Namespace) -> dict:
    fmt = "csv" if args.csv else ("json" if args.json else "html")
    return {
        "command": "report",
        "format":  fmt,
        "retries": args.retries,
    }


def dispatch(args: argparse.Namespace) -> dict:
    if args.command == "process":
        return handle_process(args)
    if args.command == "batch":
        return handle_batch(args)
    if args.command == "report":
        return handle_report(args)
    raise ValueError(f"Unknown command: {args.command}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — protected so imports don't trigger execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    result = dispatch(args)
    print(result)
