
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file into a dataframe
df = pd.read_csv('C:\codebase\FinBytes\expenses.csv')

# Calculate total expenses for each category and add more if necessary
groceries = df[df['Category'] == 'Groceries']['Amount'].sum()
utilities = df[df['Category'] == 'Utilities']['Amount'].sum()
entertainment = df[df['Category'] == 'Entertainment']['Amount'].sum()

# Create a bar chart to visualize expenses
categories = ['Groceries', 'Utilities', 'Entertainment']
amounts = [groceries, utilities, entertainment]
plt.bar(categories, amounts)
plt.xlabel('Categories')
plt.ylabel('Amount (in USD)')
plt.title('Monthly Expenses')
plt.show()


