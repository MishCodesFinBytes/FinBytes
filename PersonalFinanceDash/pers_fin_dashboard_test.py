import subprocess
import sys
import tempfile
import os


CSV_CONTENT = """date,description,amount,category
2023-02-10,Coffee,-3.50,Food
2023-02-14,Salary,2000,Income
2023-03-01,Rent,-800,Housing
2023-03-05,Groceries,-120,Food
2023-03-10,Utilities,-60,Utilities
"""


def create_temp_csv():
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    f.write(CSV_CONTENT.encode("utf-8"))
    f.close()
    return f.name


def run_command(args):
    result = subprocess.run(
        [sys.executable, "pers_fin_dashboard.py"] + args,
        capture_output=True,
        text=True
    )
    return result.stdout


def test_total_summary():
    path = create_temp_csv()
    output = run_command(["--file", path, "--summary"])
    os.unlink(path)
    assert "Food" in output and "Income" in output


def test_month_summary():
    path = create_temp_csv()
    output = run_command(["--file", path, "--month", "2023-03"])
    os.unlink(path)
    assert "Housing" in output and "Food" in output


def test_no_flag():
    path = create_temp_csv()
    output = run_command(["--file", path])
    os.unlink(path)
    assert "Please provide" in output