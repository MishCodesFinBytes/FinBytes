import math

# Function to calculate monthly payments
def monthly_payment(loan_amount, interest_rate, loan_term):
    """
    Calculate the monthly payment for a loan.
    :param loan_amount: Total loan amount
    :param interest_rate: Annual interest rate (percentage)
    :param loan_term: Loan term in years
    :return: Monthly payment amount
    """
    monthly_rate = interest_rate / (12 * 100)  # Convert annual rate to monthly rate
    num_payments = loan_term * 12  # Total number of monthly payments
    return (loan_amount * monthly_rate) / (1 - math.pow(1 + monthly_rate, -num_payments))

# Function to calculate total interest paid
def total_interest(loan_amount, interest_rate, loan_term):
    """
    Calculate total interest paid over the loan term.
    """
    monthly_pay = monthly_payment(loan_amount, interest_rate, loan_term)
    total_paid = monthly_pay * loan_term * 12
    return total_paid - loan_amount

# Function to calculate remaining balance at a given time
def remaining_balance(loan_amount, interest_rate, loan_term, months_paid):
    """
    Calculate the remaining balance after a specific number of payments.
    :param loan_amount: Total loan amount
    :param interest_rate: Annual interest rate (percentage)
    :param loan_term: Loan term in years
    :param months_paid: Number of payments made
    :return: Remaining balance
    """
    monthly_rate = interest_rate / (12 * 100)
    num_payments = loan_term * 12
    monthly_pay = monthly_payment(loan_amount, interest_rate, loan_term)
    return loan_amount * math.pow(1 + monthly_rate, months_paid) - \
           (monthly_pay / monthly_rate) * (math.pow(1 + monthly_rate, months_paid) - 1)

# Function to compare different loan terms
def compare_loan_terms(loan_amount, interest_rate, loan_terms):
    """
    Compare loan terms with their monthly payments and total interest.
    """
    print(f"\nComparison of Loan Terms for Loan Amount: ${loan_amount} at {interest_rate}% Interest Rate")
    print("-" * 70)
    print("{:<15} {:<20} {:<20}".format("Loan Term (years)", "Monthly Payment ($)", "Total Interest ($)"))
    print("-" * 70)
    for term in loan_terms:
        monthly_pay = monthly_payment(loan_amount, interest_rate, term)
        total_int = total_interest(loan_amount, interest_rate, term)
        print(f"{term:<15} {monthly_pay:<20.2f} {total_int:<20.2f}")
    print("-" * 70)

# Example inputs
loan_amount = 250000  # Loan amount in dollars
interest_rate = 3.5   # Annual interest rate in percentage
loan_term = 30        # Loan term in years
months_paid = 60      # Number of months the loan has been paid
loan_terms = [15, 20, 25, 30]  # Different loan terms to compare

# Outputs
monthly_pay = monthly_payment(loan_amount, interest_rate, loan_term)
total_int = total_interest(loan_amount, interest_rate, loan_term)
remaining_bal = remaining_balance(loan_amount, interest_rate, loan_term, months_paid)

print(f"Monthly Payment for a {loan_term}-year loan: ${monthly_pay:.2f}")
print(f"Total Interest Paid over {loan_term} years: ${total_int:.2f}")
print(f"Remaining Balance after {months_paid} months: ${remaining_bal:.2f}")

# Compare different loan terms
compare_loan_terms(loan_amount, interest_rate, loan_terms)
