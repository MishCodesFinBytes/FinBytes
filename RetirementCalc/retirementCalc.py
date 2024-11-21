import pandas as pd
import random

# --- User Inputs ---
print("### Retirement Calculator Inputs ###")
current_age = int(input("Enter your current age: "))
retirement_age = int(input("Enter the age you wish to retire: "))
annual_contribution = float(input("Enter your annual contribution (in USD): "))
desired_income = float(input("Enter your desired annual retirement income (in USD): "))
expected_return = float(input("Enter your annual return rate (in %): ")) / 100
inflation_rate = float(input("Enter the annual inflation rate (in %): ")) / 100
salary_increase_rate = float(input("Enter your annual salary increase (in %): ")) / 100
tax_rate = float(input("Enter the tax rate on retirement income (in %): ")) / 100

# Load historical stock market returns
stock_data = pd.read_csv(r"C:\codebase\FinBytes\stock_data.csv")  # Example file: "Year,Annual Return"

# Calculate years to retirement
years_to_retirement = retirement_age - current_age


# --- Variation 1: Simple Retirement Calculator ---
def simple_calculator():
    """
    Basic retirement savings calculation using fixed inputs.
    """
    required_savings = desired_income * ((1 + expected_return) ** years_to_retirement - 1) / expected_return
    print("\n### Simple Retirement Plan ###")
    print(f"Years to retirement: {years_to_retirement}")
    print(f"Total savings required: ${round(required_savings, 2)}")


# --- Variation 2: Enhanced Retirement Calculator ---
def enhanced_calculator():
    """
    Enhanced calculation that accounts for inflation, salary growth, and taxes.
    """
    total_savings_required = 0
    income = desired_income
    contribution = annual_contribution

    for _ in range(years_to_retirement):
        income *= (1 + inflation_rate)  # Adjust for inflation
        contribution *= (1 + salary_increase_rate)  # Adjust for salary growth
        yearly_savings = (income - contribution) / (1 + expected_return)
        total_savings_required += yearly_savings

    # Adjust for taxes
    total_savings_required /= (1 - tax_rate)

    print("\n### Enhanced Retirement Plan ###")
    print(f"Years to retirement: {years_to_retirement}")
    print(f"Total savings required (adjusted for inflation and taxes): ${round(total_savings_required, 2)}")


# --- Variation 3: Data-Driven Retirement Calculator ---
def data_driven_calculator():
    """
    Monte Carlo simulation using historical stock market data.
    """
    required_savings = []

    for _ in range(1000):  # Run 1000 Monte Carlo simulations
        future_returns = [random.choice(stock_data["Annual Return"]) / 100 for _ in range(years_to_retirement)]
        compounded_growth = 1
        for r in future_returns:
            compounded_growth *= (1 + r)
        savings = desired_income / compounded_growth
        required_savings.append(savings)

    # Calculate the 95th percentile savings requirement
    required_savings.sort()
    confidence_level_95 = required_savings[int(0.95 * len(required_savings))]

    print("\n### Data-Driven Retirement Plan ###")
    print(f"Years to retirement: {years_to_retirement}")
    print(f"To retire with 95% confidence, save: ${round(confidence_level_95, 2)}")


# --- Run All Calculators ---
simple_calculator()
enhanced_calculator()
data_driven_calculator()
