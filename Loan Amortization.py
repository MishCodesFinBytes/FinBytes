# Import necessary libraries
import numpy as np
import pandas as pd
import numpy_financial as npf  # Import numpy-financial for ipmt, ppmt, and pmt

'''
------------------------
Using Numpy, Pandas, and Numpy-Financial
------------------------
This section demonstrates how to use numpy, pandas, and numpy-financial 
to calculate a loan amortization schedule, showing the breakdown of 
interest and principal payments over the term of the loan.
'''

# Define loan details
loan_amount = 10000  # Total loan amount in currency
interest_rate = 0.05  # Annual interest rate (5%)
loan_term = 5  # Loan term in years

# Calculate the monthly interest rate
monthly_rate = interest_rate / 12  # Monthly interest rate (annual rate / 12)
nper = loan_term * 12  # Total number of payments (loan term in months)

# Calculate the monthly payment using the loan amortization formula
# Formula: M = P * [r(1 + r)^n] / [(1 + r)^n - 1]
# M = monthly payment, P = loan amount, r = monthly interest rate, n = number of periods
monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** nper) / ((1 + monthly_rate) ** nper - 1)

# Generate the payment number sequence (from 1 to total number of months)
payment_number = np.arange(1, nper + 1)

# Calculate interest paid each month using numpy-financial's npf.ipmt function
interest_paid = npf.ipmt(monthly_rate, payment_number, nper, loan_amount)

# Calculate principal paid each month using numpy-financial's npf.ppmt function
principal_paid = npf.ppmt(monthly_rate, payment_number, nper, loan_amount)

# Create a DataFrame using Pandas to display the loan amortization table
df = pd.DataFrame({
    'Payment Number': payment_number,  # Payment number (month)
    'Payment Amount': np.repeat(monthly_payment, nper),  # Same monthly payment for all months
    'Interest Paid': interest_paid,  # Interest paid in each month
    'Principal Paid': principal_paid  # Principal paid in each month
})

# Display the loan amortization table without index
print(df.to_string(index=False))

'''
------------------------
Using Numpy-Financial's pmt function
------------------------
This section shows how to calculate the monthly payment using numpy-financial's `pmt` function.
'''

# Define principal, annual interest rate, and loan term
principal = 10000  # Principal amount
annual_interest_rate = 5  # Annual interest rate (5%)
term_years = 5  # Loan term in years

# Calculate monthly payment using numpy-financial's pmt function
monthly_payment = npf.pmt(annual_interest_rate / 100 / 12, term_years * 12, -principal)
print(f"Monthly Payment (using numpy-financial): {monthly_payment}")

'''
------------------------
Using Pure Python for Amortization Calculation
------------------------
This section calculates the amortization schedule using pure Python without external libraries.
'''

def calculate_amortization_schedule(principal, annual_interest_rate, term_years):
    # Convert annual interest rate to monthly rate
    monthly_interest_rate = annual_interest_rate / 100 / 12
    num_payments = term_years * 12  # Total number of payments (loan term in months)
    
    # Calculate the fixed monthly payment using the loan amortization formula
    monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / ((1 + monthly_interest_rate)**num_payments - 1)
    
    # Initialize remaining loan balance
    balance = principal
    schedule = []  # List to store each month's details
    
    # Loop through each payment and calculate the breakdown of principal and interest
    for i in range(1, num_payments + 1):
        interest = balance * monthly_interest_rate  # Interest portion of the payment
        principal_payment = monthly_payment - interest  # Principal portion of the payment
        balance -= principal_payment  # Decrease the balance by principal paid
        schedule.append((i, principal_payment, interest, balance if balance > 0 else 0))  # Store month data
    
    return schedule

# Example usage of the function
amortization_schedule = calculate_amortization_schedule(principal, annual_interest_rate, term_years)
print(f"Amortization Schedule (using pure Python): {amortization_schedule[:3]}...")  # Print first 3 months for brevity

'''
------------------------
Using SymPy for Symbolic Computation
------------------------
This section shows how to solve for the monthly payment symbolically using SymPy.
'''

from sympy import symbols, Eq, solve

# Define variables for principal (P), interest rate (r), term (n), and monthly payment (M)
P, r, n, M = symbols('P r n M')

# Set up the amortization formula as an equation to solve for the monthly payment (M)
equation = Eq(M, P * r * (1 + r)**n / ((1 + r)**n - 1))

# Solve for the monthly payment (M)
monthly_payment_sympy = solve(equation, M)[0]
print(f"Monthly Payment (using SymPy): {monthly_payment_sympy}")

'''
------------------------
Using scipy.optimize to Solve for Payment Values
------------------------
This section uses scipy's fsolve function to numerically solve for the monthly payment value.
'''

from scipy.optimize import fsolve

# Define a function for loan balance after a certain number of payments
def loan_balance(monthly_payment):
    balance = principal
    for i in range(term_years * 12):
        interest = balance * (annual_interest_rate / 100 / 12)  # Interest for the current month
        balance += interest - monthly_payment  # Decrease the balance by the monthly payment
    return balance  # Return remaining loan balance after all payments

# Use fsolve to find the monthly payment that results in a zero balance at the end of the loan term
monthly_payment_scipy = fsolve(loan_balance, 200)[0]  # Initial guess is 200
print(f"Monthly Payment (using scipy.optimize): {monthly_payment_scipy}")
