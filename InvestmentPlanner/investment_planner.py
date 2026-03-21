"""
Investment Planner (CLI)

Simulates:
- Monthly contributions
- Compound growth
- Goal tracking

Run:
------
python demo.py --monthly 500 --rate 8 --years 10
python demo.py --monthly 300 --rate 7 --target 100000
"""

import argparse


# ----------------------------
# Core Calculation
# ----------------------------

def future_value(monthly, annual_rate, years):
    """
    Calculate future value of monthly investments
    """
    monthly_rate = annual_rate / 100 / 12
    months = years * 12

    total = 0

    for _ in range(months):
        total = (total + monthly) * (1 + monthly_rate)

    return round(total, 2)


def months_to_target(monthly, annual_rate, target):
    """
    Calculate months needed to reach target
    """
    monthly_rate = annual_rate / 100 / 12
    total = 0
    months = 0

    while total < target:
        total = (total + monthly) * (1 + monthly_rate)
        months += 1

        if months > 1000:
            raise ValueError("Target too high or contribution too low")

    return months, round(total, 2)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Investment Planner"
    )

    parser.add_argument("--monthly", type=float, required=True,
                        help="Monthly investment amount")
    parser.add_argument("--rate", type=float, required=True,
                        help="Annual return rate (%)")

    parser.add_argument("--years", type=int,
                        help="Investment duration (years)")
    parser.add_argument("--target", type=float,
                        help="Target investment amount")

    args = parser.parse_args()

    print("\nMonthly Investment:", args.monthly)
    print("Annual Return:", args.rate, "%")

    # Mode 1: fixed duration
    if args.years:
        total = future_value(args.monthly, args.rate, args.years)

        print("\nYears:", args.years)
        print("Final Value:", total)
        print("Total Invested:", args.monthly * args.years * 12)

    # Mode 2: goal-based
    elif args.target:
        months, total = months_to_target(
            args.monthly,
            args.rate,
            args.target
        )

        print("\nTarget:", args.target)
        print("Months to Reach Goal:", months)
        print("Years:", round(months / 12, 1))
        print("Final Value:", total)

    else:
        print("\nPlease provide either --years or --target")


if __name__ == "__main__":
    main()