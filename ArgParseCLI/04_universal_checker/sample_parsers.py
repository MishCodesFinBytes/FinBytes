"""
sample_parsers.py — Example parsers used to demonstrate check_parser_universal.py

Contains intentionally broken parsers (for the checker to catch) and
correct parsers (that should pass).

Run the checker against this file:
    python check_parser_universal.py sample_parsers.broken_parser
    python check_parser_universal.py sample_parsers.clean_parser
"""

import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Broken parser — contains all three issue types
# ─────────────────────────────────────────────────────────────────────────────

def broken_parser() -> argparse.ArgumentParser:
    """
    Parser with four deliberate issues:
      1. --verbose: store_true with default=True (Catch-22)
      2. --force:   store_false with default=False (Catch-22)
      3. --enabled: type=bool (users can't express False)
      4. --mode:    required=True + default="dev" (contradiction)
    """
    parser = argparse.ArgumentParser(description="Broken demo parser")

    # Catch-22: store_true with default=True
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output (BROKEN: always True)",
    )
    # Catch-22: store_false with default=False
    parser.add_argument(
        "--force",
        action="store_false",
        dest="force",
        default=False,
        help="Force run (BROKEN: always False)",
    )
    # type=bool misuse
    parser.add_argument(
        "--enabled",
        type=bool,
        help="Feature toggle (BROKEN: bool('false') is True)",
    )
    # required + default
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        required=True,
        default="dev",
        help="Run mode (BROKEN: required but has a default)",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Clean parser — should pass all checks
# ─────────────────────────────────────────────────────────────────────────────

def clean_parser() -> argparse.ArgumentParser:
    """
    Correctly written parser:
      - store_true with no explicit default (defaults to False)
      - store_false with no explicit default (defaults to True)
      - BooleanOptionalAction for symmetric control
      - required without a default
      - meaningful choices
    """
    parser = argparse.ArgumentParser(description="Clean demo parser")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable caching",
    )
    parser.add_argument(
        "--feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the feature",
    )
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Target environment",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path (required, no default)",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Parser with subcommands — checker should recurse into subparsers
# ─────────────────────────────────────────────────────────────────────────────

def subcommand_parser_with_issues() -> argparse.ArgumentParser:
    """Parser with a broken flag inside a subcommand."""
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")

    run_parser = subs.add_parser("run")
    run_parser.add_argument(
        "--debug",
        action="store_true",
        default=True,   # Catch-22 inside subcommand
        help="Debug mode (BROKEN)",
    )
    return parser
