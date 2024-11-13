'''
Explanation:
1) Expense Calculation: It prompts the user for 3 expense entries and checks if the total exceeds 
the set budget.
2) Savings Goal Calculation: It calculates 20% of the monthly income and prints the savings target.
3) Loan Interest Calculation: It calculates the total interest on a loan of $10,000 over 5 years at 
8% annual interest using the compound interest formula.
'''


# --------------------------
# Calculating Expenses
# --------------------------

# Set the budget limit
budget = 2000

# Create an empty list to store the expenses
expenses = []

# Collect expenses from the user (3 entries)
for i in range(3):
    expense = float(input("Enter expense: "))  # Input expense value
    expenses.append(expense)  # Add the expense to the list

# Calculate the total expenses
total_expenses = sum(expenses)

# Check if total expenses exceed the budget
if total_expenses > budget:
    print("You have exceeded your budget!")
else:
    print("You have spent within your budget.")


# --------------------------
# Calculating Savings
# --------------------------

# Set the monthly income
monthly_income = 5000

# Calculate the savings goal (20% of the monthly income)
savings_goal = 0.20 * monthly_income

# Print the savings goal
print("To reach your savings goal, you need to save $" + str(savings_goal) + " each month.")


# --------------------------
# Calculating Loan Interest
# --------------------------

# Set the loan amount and interest rate
loan_amount = 10000
interest_rate = 0.08

# Calculate the total interest over 5 years using compound interest formula
total_interest = loan_amount * ((1 + interest_rate) ** 5 - 1)

# Print the total interest to be paid over 5 years
print("You will pay a total of ${:.2f} in interest over 5 years.".format(total_interest))