"""
Credit Card Monthly Payment Calculator

Based on:
Calculate Monthly Payments for Credit Card Debt

Run:
------
python demo.py --balance 10000 --rate 18 --months 36
"""

import argparse


# ----------------------------
# Core Calculation
# ----------------------------

def calculate_monthly_payment(balance, annual_rate, months):
    """
    Calculate monthly payment using amortisation formula
    """
    monthly_rate = annual_rate / 100 / 12

    if monthly_rate == 0:
        return round(balance / months, 2)

    payment = (
        balance * monthly_rate * (1 + monthly_rate) ** months
    ) / ((1 + monthly_rate) ** months - 1)

    return round(payment, 2)


def calculate_totals(balance, payment, annual_rate):
    """
    Simulate payoff to compute total interest
    """
    monthly_rate = annual_rate / 100 / 12

    total_paid = 0
    months = 0

    while balance > 0:
        interest = balance * monthly_rate
        balance += interest
        balance -= payment

        total_paid += payment
        months += 1

        # safety break
        if months > 600:
            raise ValueError("Payment too low to ever repay debt")

    return round(total_paid, 2), months


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Payment Calculator"
    )

    parser.add_argument("--balance", type=float, required=True)
    parser.add_argument("--rate", type=float, required=True,
                        help="Annual interest rate (e.g. 18)")
    parser.add_argument("--months", type=int, required=True)

    args = parser.parse_args()

    payment = calculate_monthly_payment(
        args.balance,
        args.rate,
        args.months
    )

    total_paid, actual_months = calculate_totals(
        args.balance,
        payment,
        args.rate
    )

    print("\nBalance:", args.balance)
    print("Interest Rate:", args.rate, "%")
    print("Planned Months:", args.months)

    print("\nMonthly Payment:", payment)
    print("Total Paid:", total_paid)
    print("Interest Paid:", round(total_paid - args.balance, 2))
    print("Actual Months to Payoff:", actual_months)


if __name__ == "__main__":
    main()
