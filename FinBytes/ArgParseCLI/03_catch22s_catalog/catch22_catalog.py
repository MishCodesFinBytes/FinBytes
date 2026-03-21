"""
catch22_catalog.py — Runnable demo for:
"Argparse Catch-22s: When Your CLI Accepts Flags That Can't Do Anything"

All 9 Catch-22 patterns from the blog catalogued as runnable functions,
each paired with the broken version and the correct fix.

Run:
    python catch22_catalog.py
    pytest test_catch22_catalog.py -v
"""

import argparse
import os


# ─────────────────────────────────────────────────────────────────────────────
# Utility: check whether an argument can change its own value
# ─────────────────────────────────────────────────────────────────────────────

def can_flag_change_value(
    parser: argparse.ArgumentParser,
    dest: str,
    flag: str,
    flag_value: str | None = None,
) -> bool:
    """Returns True if the flag being present changes the resulting value."""
    absent_args = []
    present_args = [flag] if flag_value is None else [flag, flag_value]
    val_absent  = getattr(parser.parse_args(absent_args),  dest)
    val_present = getattr(parser.parse_args(present_args), dest)
    return val_absent != val_present


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #1: Boolean flags that cannot change state
# ─────────────────────────────────────────────────────────────────────────────

def broken_1_store_true_default_true() -> argparse.ArgumentParser:
    """store_true + default=True → value is always True."""
    p = argparse.ArgumentParser()
    p.add_argument("--feature", action="store_true", default=True)
    return p


def fixed_1_store_true() -> argparse.ArgumentParser:
    """FIX: Remove the default — store_true already defaults to False."""
    p = argparse.ArgumentParser()
    p.add_argument("--feature", action="store_true")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #2: Mutually exclusive groups that aren't real choices
# ─────────────────────────────────────────────────────────────────────────────

def broken_2_exclusive_group_fake_choice() -> argparse.ArgumentParser:
    """--fast with default=True is a no-op alongside --slow."""
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true", default=True)
    group.add_argument("--slow", action="store_false", dest="fast")
    return p


def fixed_2_boolean_optional_action() -> argparse.ArgumentParser:
    """FIX: BooleanOptionalAction creates --fast and --no-fast symmetrically."""
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #3: required=True + default — contradictory signals
# ─────────────────────────────────────────────────────────────────────────────

def broken_3_required_with_default() -> argparse.ArgumentParser:
    """required=True with a default — help says required but it never fails."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "prod"], required=True, default="dev")
    return p


def fixed_3a_required_no_default() -> argparse.ArgumentParser:
    """FIX option A: required=True, no default — user must always specify."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "prod"], required=True)
    return p


def fixed_3b_optional_with_default() -> argparse.ArgumentParser:
    """FIX option B: optional with default — user can override, not required."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "prod"], default="dev")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #4: choices list collapsed to a single value
# ─────────────────────────────────────────────────────────────────────────────

def broken_4_single_choice() -> argparse.ArgumentParser:
    """choices=["prod"] with default="prod" — user input changes nothing."""
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["prod"], default="prod")
    return p


def fixed_4_meaningful_choices() -> argparse.ArgumentParser:
    """FIX: Restore meaningful alternatives."""
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["dev", "staging", "prod"], default="dev")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #5: nargs="?" with identical default and const
# ─────────────────────────────────────────────────────────────────────────────

def broken_5_nargs_identical_default_const() -> argparse.ArgumentParser:
    """default=1 and const=1 — every usage path gives 1."""
    p = argparse.ArgumentParser()
    p.add_argument("--level", nargs="?", default=1, const=1)
    return p


def fixed_5_nargs_distinct_values() -> argparse.ArgumentParser:
    """FIX: const differs from default so bare --level means something."""
    p = argparse.ArgumentParser()
    p.add_argument("--level", nargs="?", default=1, const=2)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #6: mutable default list (shared state across parses)
# ─────────────────────────────────────────────────────────────────────────────

def broken_6_mutable_default() -> argparse.ArgumentParser:
    """default=[] is shared — state leaks between parse calls."""
    p = argparse.ArgumentParser()
    p.add_argument("--tag", action="append", default=[])
    return p


def fixed_6_none_default() -> argparse.ArgumentParser:
    """FIX: Use default=None and normalize after parsing."""
    p = argparse.ArgumentParser()
    p.add_argument("--tag", action="append", default=None)
    return p


def normalize_tags(args: argparse.Namespace) -> list[str]:
    return args.tag if args.tag is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #7: env var silently overrides CLI flag
# ─────────────────────────────────────────────────────────────────────────────

def broken_7_env_silently_wins(cli_timeout: int | None = None) -> int:
    """
    CLI flag is silently overridden by env var — user cannot enforce CLI value.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args(["--timeout", str(cli_timeout)] if cli_timeout else [])
    # BUG: env var always wins, CLI flag is ignored
    return int(os.getenv("TIMEOUT", args.timeout))


def fixed_7_explicit_precedence(cli_timeout: int | None = None) -> dict:
    """FIX: Document and enforce explicit precedence."""
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=int, default=None)
    args = p.parse_args(["--timeout", str(cli_timeout)] if cli_timeout else [])

    # Explicit precedence: CLI > env > built-in default
    timeout = args.timeout or int(os.getenv("TIMEOUT", 30))
    return {"timeout": timeout, "source": "cli" if args.timeout else "env_or_default"}


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #8: type=bool — impossible to express False via CLI
# ─────────────────────────────────────────────────────────────────────────────

def broken_8_type_bool() -> argparse.ArgumentParser:
    """type=bool — bool("false") is True, so --enabled false enables it."""
    p = argparse.ArgumentParser()
    p.add_argument("--enabled", type=bool)
    return p


def fixed_8_store_true_store_false() -> argparse.ArgumentParser:
    """FIX: Use store_true/store_false instead of type=bool."""
    p = argparse.ArgumentParser()
    p.add_argument("--enable",  action="store_true",  dest="enabled")
    p.add_argument("--disable", action="store_false", dest="enabled")
    p.set_defaults(enabled=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Catch-22 #9: set_defaults bypasses mutually exclusive group
# ─────────────────────────────────────────────────────────────────────────────

def broken_9_set_defaults_bypass() -> argparse.ArgumentParser:
    """set_defaults fixes the outcome — group implies choice that doesn't exist."""
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true")
    group.add_argument("--yaml", action="store_true")
    p.set_defaults(json=True)
    return p


def fixed_9_explicit_format_choice() -> argparse.ArgumentParser:
    """FIX: Use a single --format argument with choices — unambiguous."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--format",
        choices=["json", "yaml"],
        required=True,
        help="Output format",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Pre-commit checklist function
# ─────────────────────────────────────────────────────────────────────────────

def audit_parser(parser: argparse.ArgumentParser) -> list[str]:
    """
    Quick audit of a parser: returns a list of detected Catch-22 issues.
    Checks: boolean no-op flags, required+default, single choices.
    """
    issues = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        flag = action.option_strings[0]

        # Boolean no-op
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            try:
                absent  = getattr(parser.parse_args([]),      action.dest)
                present = getattr(parser.parse_args([flag]),   action.dest)
                if absent == present:
                    issues.append(f"[CATCH-22] {flag} cannot change value (always {absent})")
            except SystemExit:
                pass

        # required + default
        if getattr(action, "required", False) and action.default is not None:
            issues.append(f"[REQUIRED+DEFAULT] {flag} is required but has default={action.default!r}")

        # single-value choices
        if action.choices and len(action.choices) == 1:
            issues.append(f"[SINGLE-CHOICE] {flag} choices={list(action.choices)} — only one option available")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Argparse Catch-22 Catalog ===\n")

    checks = [
        ("Catch-22 #1: store_true default=True", broken_1_store_true_default_true(), "feature", "--feature"),
        ("Catch-22 #4: single choices value",     broken_4_single_choice(),           "env",     "--env"),
    ]
    for label, parser, dest, flag in checks:
        try:
            changed = can_flag_change_value(parser, dest, flag)
        except SystemExit:
            changed = None
        status = "✅ can change" if changed else "❌ cannot change (Catch-22)"
        print(f"  {label}: {status}")

    print("\n--- Catch-22 #3: required + default ---")
    p = broken_3_required_with_default()
    try:
        args = p.parse_args([])  # should fail but doesn't due to default
        print(f"  parse_args([]) succeeded (mode={args.mode!r}) — required was bypassed by default")
    except SystemExit:
        print("  Correctly raised SystemExit")

    print("\n--- Catch-22 #5: nargs='?' identical default/const ---")
    p = broken_5_nargs_identical_default_const()
    absent  = p.parse_args([]).level
    bare    = p.parse_args(["--level"]).level
    explicit= p.parse_args(["--level", "1"]).level
    print(f"  absent={absent}  --level={bare}  --level 1={explicit}  (all same: {absent==bare==explicit})")

    print("\n--- Catch-22 #8: type=bool ---")
    p = broken_8_type_bool()
    val = p.parse_args(["--enabled", "false"]).enabled
    print(f"  --enabled false → {val}  (should be False, but is {val})")

    print("\n--- Audit function ---")
    issues = audit_parser(broken_1_store_true_default_true())
    for issue in issues:
        print(f"  {issue}")

    print()


if __name__ == "__main__":
    main()
