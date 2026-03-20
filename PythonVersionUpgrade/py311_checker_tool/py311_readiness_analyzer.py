"""
py311_readiness_analyzer.py — Runnable demo for:
"py36 to py311 migration readiness checker tool"

Static analyzer that scans .py files for common Python 3.6 → 3.11
migration pitfalls. Produces a readable report and exits with code 1
if issues are found (CI-friendly).

Usage:
    python py311_readiness_analyzer.py my_script.py
    python py311_readiness_analyzer.py ./my_project/

Output:
    Python 3.11 readiness issues found (5):
      - my_script.py:42 - unsafe None comparison ...
      - my_script.py:108 - check date/datetime arithmetic
      ...
"""

import ast
import sys
import os


class Py311Analyzer(ast.NodeVisitor):
    """
    Static analyzer for Python 3.11 readiness.
    Detects common migration issues from Python 3.6 → 3.11.

    Each visit_* method corresponds to one category of issue.
    Extend by adding more visit_* methods as needed.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: list[str] = []

    # ─────────────────────────────────────────────────────────────────────
    # 1. Unsafe None comparisons: <, >, <=, >=
    #    These raised TypeError in 3.11; 3.6 sometimes silently passed.
    # ─────────────────────────────────────────────────────────────────────

    def visit_Compare(self, node: ast.Compare) -> None:
        ordering_ops = (ast.Lt, ast.Gt, ast.LtE, ast.GtE)
        for op in node.ops:
            if isinstance(op, ordering_ops):
                all_operands = [node.left] + node.comparators
                for operand in all_operands:
                    if isinstance(operand, ast.Constant) and operand.value is None:
                        self.issues.append(
                            f"{self.filename}:{node.lineno} - "
                            f"[None comparison] ordering comparison with None raises TypeError in 3.11"
                        )
        self.generic_visit(node)

    # ─────────────────────────────────────────────────────────────────────
    # 2. Date/datetime arithmetic
    #    Mixing date and datetime in arithmetic fails in 3.11.
    # ─────────────────────────────────────────────────────────────────────

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            left_name = getattr(node.left, "id", "") or ""
            right_name = getattr(node.right, "id", "") or ""
            if "date" in left_name.lower() or "date" in right_name.lower():
                self.issues.append(
                    f"{self.filename}:{node.lineno} - "
                    f"[date arithmetic] check date/datetime arithmetic — mixing types raises TypeError in 3.11"
                )
        self.generic_visit(node)

    # ─────────────────────────────────────────────────────────────────────
    # 3. Pandas .dt accessor
    #    Using .dt on a non-datetime column now raises in 3.11+ pandas.
    # ─────────────────────────────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "dt":
            self.issues.append(
                f"{self.filename}:{node.lineno} - "
                f"[pandas .dt] ensure column is explicitly cast to datetime before .dt accessor"
            )
        self.generic_visit(node)

    # ─────────────────────────────────────────────────────────────────────
    # 4. Ambiguous truthiness on common DataFrame/array variable names
    #    `if df:` raises ValueError in pandas; `if array:` is ambiguous.
    # ─────────────────────────────────────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> None:
        ambiguous_names = {"df", "series", "array", "data", "frame", "result"}
        if node.id.lower() in ambiguous_names:
            # Only flag if this Name appears as the test of an If node
            # (We track context via parent; simplified here to flag all occurrences)
            self.issues.append(
                f"{self.filename}:{node.lineno} - "
                f"[ambiguous truthiness] avoid bare `if {node.id}:` — use `.empty`, `is None`, or `len()`"
            )
        self.generic_visit(node)

    # ─────────────────────────────────────────────────────────────────────
    # 5. String/bytes mixing via print, input, open
    #    These surface implicit encoding issues that 3.11 exposes earlier.
    # ─────────────────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in {"print", "input", "open"}:
            self.issues.append(
                f"{self.filename}:{node.lineno} - "
                f"[string/bytes] check `{func_name}()` for implicit string/bytes mixing or encoding assumptions"
            )
        self.generic_visit(node)

    # ─────────────────────────────────────────────────────────────────────
    # 6. `is` used for string/value equality
    #    3.6 interning sometimes made `is` work; 3.11 is less forgiving.
    # ─────────────────────────────────────────────────────────────────────

    def visit_Compare_is(self, node: ast.Compare) -> None:
        # Handled inside visit_Compare to avoid double registration
        pass

    def _check_is_equality(self, node: ast.Compare) -> None:
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, ast.Is):
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    self.issues.append(
                        f"{self.filename}:{node.lineno} - "
                        f"[is vs ==] use `==` for string value comparison, not `is`"
                    )

    # Override visit_Compare to also call _check_is_equality
    _original_visit_Compare = visit_Compare

    def visit_Compare(self, node: ast.Compare) -> None:  # type: ignore[override]
        self._original_visit_Compare(node)
        self._check_is_equality(node)


# ─────────────────────────────────────────────────────────────────────────────
# File and directory analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(filepath: str) -> list[str]:
    """Parse and analyze a single .py file. Returns a list of issue strings."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        return [f"{filepath} - [read error] {e}"]

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return [f"{filepath} - [parse error] {e}"]

    analyzer = Py311Analyzer(filepath)
    analyzer.visit(tree)
    return analyzer.issues


def analyze_path(path: str) -> list[str]:
    """Analyze a single file or recursively walk a directory."""
    all_issues: list[str] = []

    if os.path.isfile(path):
        if path.endswith(".py"):
            all_issues.extend(analyze_file(path))
        else:
            print(f"Skipping non-Python file: {path}")

    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in sorted(files):
                if filename.endswith(".py"):
                    full_path = os.path.join(root, filename)
                    all_issues.extend(analyze_file(full_path))
    else:
        print(f"Path not found: {path}")

    return all_issues


def print_report(issues: list[str], path: str) -> None:
    """Print a readable report to stdout."""
    if not issues:
        print(f"\n✅  No Python 3.11 migration issues detected in: {path}\n")
        return

    print(f"\n⚠️   Python 3.11 readiness issues found ({len(issues)}):\n")
    for issue in issues:
        print(f"  - {issue}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python py311_readiness_analyzer.py <file_or_directory>")
        print("       python py311_readiness_analyzer.py ./my_project/")
        return 2

    path = sys.argv[1]
    issues = analyze_path(path)
    print_report(issues, path)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
