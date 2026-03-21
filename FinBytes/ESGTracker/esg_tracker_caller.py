import subprocess
import sys


def run_command(args):
    result = subprocess.run(
        [sys.executable, "C:\codebase\FinBytes\esg_tracker_demo.py"] + args,
        capture_output=True,
        text=True
    )
    return result.stdout


def test_valid_data():
    output = run_command(["--company", "AAPL", "--date", "2023-01"])
    assert "ESG Score" in output


def test_invalid_date():
    output = run_command(["--company", "AAPL", "--date", "2022-01"])
    assert "No data found" in output


def test_correct_company():
    output = run_command(["--company", "TSLA", "--date", "2023-02"])
    assert "TSLA" in output