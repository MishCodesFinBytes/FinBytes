import subprocess
import sys


def run_command(args):
    result = subprocess.run(
        [sys.executable, "cred_card_repay_calc.py"] + args,
        capture_output=True,
        text=True
    )
    return result.stdout


def test_basic_calculation():
    output = run_command([
        "--balance", "10000",
        "--rate", "18",
        "--months", "36"
    ])
    assert "Monthly Payment" in output


def test_interest_present():
    output = run_command([
        "--balance", "5000",
        "--rate", "20",
        "--months", "24"
    ])
    assert "Interest Paid" in output


def test_zero_interest():
    output = run_command([
        "--balance", "1200",
        "--rate", "0",
        "--months", "12"
    ])
    assert "Monthly Payment" in output