"""
Personal Finance Dashboard (CLI)

Processes a transactions CSV and provides summary reports.

CSV must have columns:
date, description, amount, category

Run:
------
python demo.py --file data.csv --month 2023-03
python demo.py --file data.csv --summary
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime


# ----------------------------
# Data Loading
# ----------------------------

def load_transactions(filepath):
    data = []
    with open(filepath, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row["amount"] = float(row["amount"])
            data.append(row)
    return data


# ----------------------------
# Summary Calculations
# ----------------------------

def summarize_by_category(transactions):
    totals = defaultdict(float)
    for t in transactions:
        totals[t["category"]] += t["amount"]
    return dict(totals)


def monthly_summary(transactions, month):
    monthly = [
        t for t in transactions
        if t["date"].startswith(month)
    ]
    return summarize_by_category(monthly)


def total_summary(transactions):
    return summarize_by_category(transactions)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Personal Finance Summary"
    )

    parser.add_argument("--file", required=True,
                        help="CSV file with transactions")
    parser.add_argument("--month",
                        help="Month to summarise (YYYY-MM)")
    parser.add_argument("--summary", action="store_true",
                        help="Show total summary")

    args = parser.parse_args()

    data = load_transactions(args.file)

    if args.summary:
        totals = total_summary(data)
        print("\nTotal Summary:")
        for cat, amt in totals.items():
            print(f"{cat}: {amt:.2f}")

    elif args.month:
        totals = monthly_summary(data, args.month)
        print(f"\nSummary for {args.month}:")
        for cat, amt in totals.items():
            print(f"{cat}: {amt:.2f}")

    else:
        print("Please provide either --summary or --month")


if __name__ == "__main__":
    main()
