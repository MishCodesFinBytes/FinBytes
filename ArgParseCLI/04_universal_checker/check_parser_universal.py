"""
check_parser_universal.py — The Universal Argparse Catch-22 Checker from:
"Universal Argparse Catch-22 Checker"

Detects three classes of broken argparse arguments:
  1. Boolean Catch-22 — store_true/store_false flags that cannot change value
  2. type=bool misuse — bool("false") is True, so users can't express False
  3. Required + default contradiction — required implies no default, but one exists

Usage:
    python check_parser_universal.py myapp.cli.build_parser

Pre-commit hook (.pre-commit-config.yaml):
    - repo: local
      hooks:
        - id: check-argparse
          name: Check Argparse Sanity
          entry: python check_parser_universal.py myapp.cli.build_parser
          language: system
          types: [python]

Exit codes:
    0 — all checks passed
    1 — issues found
    2 — usage error

Tests:
    pytest test_check_parser_universal.py -v
"""

import argparse
import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Issue data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    category: str
    flag: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.flag}: {self.detail}"


# ─────────────────────────────────────────────────────────────────────────────
# Core checker
# ─────────────────────────────────────────────────────────────────────────────

def check_parser(parser: argparse.ArgumentParser) -> list[Issue]:
    """
    Inspects every optional action in the parser and returns a list of Issues.
    Recurses into subparser groups automatically.
    """
    issues: list[Issue] = []
    _check_actions(parser, issues)

    # Recurse into subparsers
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                _check_actions(subparser, issues)

    return issues


def _check_actions(parser: argparse.ArgumentParser, issues: list[Issue]) -> None:
    """Checks all optional actions in a single parser namespace."""
    for action in parser._actions:
        if not action.option_strings:
            continue   # skip positional arguments

        flag = action.option_strings[0]

        # ── Check 1: Boolean Catch-22 ────────────────────────────────────────
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            try:
                val_absent  = getattr(parser.parse_args([]),      action.dest)
                val_present = getattr(parser.parse_args([flag]),   action.dest)
            except SystemExit:
                # Parser has required args — skip this check
                continue

            if val_absent == val_present:
                issues.append(Issue(
                    category="Catch-22",
                    flag=flag,
                    detail=(
                        f"cannot change value — always {val_absent}. "
                        f"action={type(action).__name__}, default={action.default!r}. "
                        f"Fix: remove the explicit default= or use BooleanOptionalAction."
                    ),
                ))

        # ── Check 2: type=bool misuse ─────────────────────────────────────────
        if action.type is bool:
            issues.append(Issue(
                category="TypeBool",
                flag=flag,
                detail=(
                    "uses type=bool — bool('false') is True, so users cannot express falsehood. "
                    "Fix: use action='store_true' or action='store_false' instead."
                ),
            ))

        # ── Check 3: required=True + default ─────────────────────────────────
        if getattr(action, "required", False) and action.default is not None:
            issues.append(Issue(
                category="RequiredWithDefault",
                flag=flag,
                detail=(
                    f"is required=True but has default={action.default!r}. "
                    f"This contradicts itself — the default silently satisfies required. "
                    f"Fix: use either required=True (no default) or a default (not required)."
                ),
            ))

        # ── Check 4: choices collapsed to a single value ─────────────────────
        if action.choices is not None and len(list(action.choices)) == 1:
            only = list(action.choices)[0]
            issues.append(Issue(
                category="SingleChoice",
                flag=flag,
                detail=(
                    f"choices has only one option ({only!r}) — user input cannot vary. "
                    f"Fix: add meaningful alternatives or remove the argument."
                ),
            ))


# ─────────────────────────────────────────────────────────────────────────────
# Loader: import a parser by dotted path
# ─────────────────────────────────────────────────────────────────────────────

def load_parser_from_path(dotted_path: str) -> argparse.ArgumentParser:
    """
    Imports a function or class by dotted path and calls it to get a parser.

    Examples:
        myapp.cli.build_parser        — calls build_parser()
        myapp.cli.ParserClass         — calls ParserClass()
    """
    module_path, func_name = dotted_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Cannot import module {module_path!r}: {e}") from e

    func_or_class = getattr(module, func_name, None)
    if func_or_class is None:
        raise AttributeError(f"{func_name!r} not found in {module_path!r}")

    if not callable(func_or_class):
        raise TypeError(f"{dotted_path!r} is not callable")

    parser = func_or_class()

    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError(
            f"{dotted_path!r} returned {type(parser).__name__!r}, "
            f"expected ArgumentParser"
        )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(issues: list[Issue], source: str) -> None:
    if not issues:
        print(f"\n✅  Parser sanity check PASSED  ({source})\n")
        return

    print(f"\n❌  Parser sanity check FAILED  ({source})")
    print(f"    {len(issues)} issue(s) found:\n")
    for issue in issues:
        print(f"  {issue}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]

    if len(argv) != 1:
        print("Usage: python check_parser_universal.py <module.path.build_parser>")
        print("       python check_parser_universal.py myapp.cli.build_parser")
        return 2

    dotted_path = argv[0]

    try:
        parser = load_parser_from_path(dotted_path)
    except (ImportError, AttributeError, TypeError) as e:
        print(f"Error loading parser: {e}")
        return 2

    issues = check_parser(parser)
    print_report(issues, dotted_path)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
