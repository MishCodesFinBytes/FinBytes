import subprocess
import sys


def run_command(args):
    result = subprocess.run(
        [sys.executable, "investment_planner.py"] + args,
        capture_output=True,
        text=True
    )
    return result.stdout


def test_future_value():
    output = run_command([
        "--monthly", "500",
        "--rate", "8",
        "--years", "5"
    ])
    assert "Final Value" in output


def test_target_mode():
    output = run_command([
        "--monthly", "300",
        "--rate", "7",
        "--target", "50000"
    ])
    assert "Months to Reach Goal" in output


def test_input_validation():
    output = run_command([
        "--monthly", "300",
        "--rate", "7"
    ])
    assert "Please provide either" in output
