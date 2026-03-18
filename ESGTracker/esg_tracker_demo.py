"""
ESG Score Tracking (Historical Data Version)

Simulates real ESG tracking using historical dataset.

Run:
------
python demo.py --company AAPL --date 2023-01
python demo.py --company TSLA --date 2023-03
"""

import argparse


# ----------------------------
# Historical ESG Dataset
# ----------------------------

ESG_DATA = [
    {"company": "AAPL", "date": "2023-01", "E": 78, "S": 72, "G": 80},
    {"company": "AAPL", "date": "2023-02", "E": 80, "S": 74, "G": 82},
    {"company": "AAPL", "date": "2023-03", "E": 77, "S": 75, "G": 81},

    {"company": "TSLA", "date": "2023-01", "E": 65, "S": 60, "G": 55},
    {"company": "TSLA", "date": "2023-02", "E": 68, "S": 62, "G": 58},
    {"company": "TSLA", "date": "2023-03", "E": 70, "S": 64, "G": 60},
]


# ----------------------------
# Data Fetch
# ----------------------------

def get_esg_record(company, date):
    for record in ESG_DATA:
        if record["company"] == company and record["date"] == date:
            return record
    return None


# ----------------------------
# ESG Score Calculation
# ----------------------------

def calculate_esg_score(record):
    return round(
        0.4 * record["E"] +
        0.3 * record["S"] +
        0.3 * record["G"], 2
    )


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Historical ESG Score Tracker"
    )

    parser.add_argument("--company", required=True)
    parser.add_argument("--date", required=True)

    args = parser.parse_args()

    record = get_esg_record(args.company, args.date)

    if not record:
        print("No data found for given company/date")
        return

    score = calculate_esg_score(record)

    print("\nCompany:", record["company"])
    print("Date:", record["date"])
    print("E:", record["E"],
          "S:", record["S"],
          "G:", record["G"])
    print("ESG Score:", score)


if __name__ == "__main__":
    main()