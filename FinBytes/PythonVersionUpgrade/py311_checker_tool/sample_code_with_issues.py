"""
sample_code_with_issues.py — Example file WITH known Python 3.6 → 3.11 issues.

Run the analyzer against this file to see it in action:
    python py311_readiness_analyzer.py sample_code_with_issues.py

You should see issues flagged for:
  - None ordering comparison
  - date arithmetic
  - pandas .dt accessor
  - ambiguous truthiness
  - string/bytes usage via open/print
"""

from datetime import date, datetime
import pandas as pd


# Issue 1: unsafe None comparison (ordering with None)
def compare_scores(score, threshold=None):
    if score > threshold:     # TypeError in 3.11 if threshold is None
        return True
    return False


# Issue 2: date/datetime arithmetic
def days_until_deadline(deadline_date):
    today = datetime.now()
    delta = today - deadline_date   # fails if deadline_date is a date, not datetime
    return delta.days


# Issue 3: pandas .dt accessor without explicit cast
def extract_year(df):
    return df["event_date"].dt.year   # may fail if column isn't datetime dtype


# Issue 4: ambiguous truthiness on a DataFrame
def process(df):
    if df:                            # raises ValueError in pandas
        return df.head()
    return None


# Issue 5: open() — potential encoding assumption
def read_config(path):
    with open(path) as f:             # no encoding= specified
        return f.read()


# Issue 6: is used for string comparison
def check_status(status):
    if status is "OK":                # works by accident in 3.6 due to interning
        return True
    return False
