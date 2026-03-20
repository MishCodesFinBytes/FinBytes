"""
boolean_catch22.py — Runnable demo for:
"The argparse Catch-22: How Boolean Flags Quietly Break Your CLI"

Demonstrates every combination from the truth table:
  - store_true with default unset  → CORRECT
  - store_true with default=False  → redundant but works
  - store_true with default=True   → BROKEN (Catch-22)
  - store_false with default unset → CORRECT
  - store_false with default=True  → redundant but works
  - store_false with default=False → BROKEN (Catch-22)
  - BooleanOptionalAction           → CORRECT symmetric control

Run:
    python boolean_catch22.py
    pytest test_boolean_catch22.py -v
"""

import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Helper: simulate "flag absent" and "flag present" for a given parser
# ─────────────────────────────────────────────────────────────────────────────

def probe(parser: argparse.ArgumentParser, dest: str, flag: str) -> dict:
    """Returns the value of `dest` when the flag is absent vs present."""
    absent  = getattr(parser.parse_args([]),     dest)
    present = getattr(parser.parse_args([flag]),  dest)
    can_change = absent != present
    return {
        "flag":       flag,
        "dest":       dest,
        "absent":     absent,
        "present":    present,
        "can_change": can_change,
        "status":     "✅ CORRECT" if can_change else "❌ CATCH-22 (broken)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# store_true combinations
# ─────────────────────────────────────────────────────────────────────────────

def build_store_true_correct() -> argparse.ArgumentParser:
    """CORRECT: store_true, no explicit default (defaults to False)."""
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true")
    return p


def build_store_true_redundant() -> argparse.ArgumentParser:
    """REDUNDANT but harmless: store_true with default=False (same as above)."""
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true", default=False)
    return p


def build_store_true_broken() -> argparse.ArgumentParser:
    """BROKEN: store_true with default=True — flag can never change value."""
    p = argparse.ArgumentParser()
    p.add_argument("--feature", action="store_true", default=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# store_false combinations
# ─────────────────────────────────────────────────────────────────────────────

def build_store_false_correct() -> argparse.ArgumentParser:
    """CORRECT: store_false, no explicit default (defaults to True)."""
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_false", dest="use_cache")
    return p


def build_store_false_redundant() -> argparse.ArgumentParser:
    """REDUNDANT but harmless: store_false with default=True (same as above)."""
    p = argparse.ArgumentParser()
    p.add_argument("--no-auth", action="store_false", dest="use_auth", default=True)
    return p


def build_store_false_broken() -> argparse.ArgumentParser:
    """BROKEN: store_false with default=False — flag can never change value."""
    p = argparse.ArgumentParser()
    p.add_argument("--no-logging", action="store_false", dest="logging", default=False)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# BooleanOptionalAction — symmetric control (the correct solution)
# ─────────────────────────────────────────────────────────────────────────────

def build_boolean_optional_action() -> argparse.ArgumentParser:
    """
    CORRECT: BooleanOptionalAction creates both --feature and --no-feature.
    Symmetric control — user can enable or disable from the CLI.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# type=bool — the impossible boolean
# ─────────────────────────────────────────────────────────────────────────────

def build_type_bool_broken() -> argparse.ArgumentParser:
    """
    BROKEN: type=bool means bool("false") == True.
    Users cannot express falsehood — every non-empty string is truthy.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--enabled", type=bool)
    return p


def demonstrate_type_bool_bug() -> dict:
    """Shows that --enabled false evaluates as True."""
    p = build_type_bool_broken()
    args_false_string = p.parse_args(["--enabled", "false"])
    args_zero_string  = p.parse_args(["--enabled", "0"])
    return {
        "--enabled false": args_false_string.enabled,   # True (bug!)
        "--enabled 0":     args_zero_string.enabled,    # True (bug!)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main demo — prints the full truth table
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Boolean Flag Catch-22 Truth Table ===\n")

    cases = [
        ("store_true (correct)",   build_store_true_correct(),   "verbose",    "--verbose"),
        ("store_true (redundant)", build_store_true_redundant(), "debug",      "--debug"),
        ("store_true (BROKEN)",    build_store_true_broken(),    "feature",    "--feature"),
        ("store_false (correct)",  build_store_false_correct(),  "use_cache",  "--no-cache"),
        ("store_false (redundant)",build_store_false_redundant(),"use_auth",   "--no-auth"),
        ("store_false (BROKEN)",   build_store_false_broken(),   "logging",    "--no-logging"),
    ]

    print(f"{'Label':<30} {'Flag':<15} {'Absent':>8} {'Present':>8} {'Status'}")
    print("─" * 85)
    for label, parser, dest, flag in cases:
        r = probe(parser, dest, flag)
        print(f"{label:<30} {flag:<15} {str(r['absent']):>8} {str(r['present']):>8}  {r['status']}")

    print("\n=== BooleanOptionalAction (symmetric) ===\n")
    p = build_boolean_optional_action()
    on  = p.parse_args(["--feature"])
    off = p.parse_args(["--no-feature"])
    dflt= p.parse_args([])
    print(f"  --feature:    {on.feature}")
    print(f"  --no-feature: {off.feature}")
    print(f"  (absent):     {dflt.feature}")

    print("\n=== type=bool bug ===\n")
    results = demonstrate_type_bool_bug()
    for cli, val in results.items():
        print(f"  {cli!r:<25} → {val}  {'(bug! expected False)' if val else ''}")

    print()


if __name__ == "__main__":
    main()
