# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# ============================
# 1. Calculating Average Stock Price
# ============================

# Load the dataset containing stock prices
df_stock_prices = pd.read_csv(r'C:\codebase\FinBytes\stock_prices.csv')

# Calculate the average stock price by finding the mean of the 'Price' column
avg_price = df_stock_prices['Price'].mean()

# Display the calculated average stock price
print("Average stock price:", avg_price)


# ==============================
# 2. Visualizing Customer Spending Patterns
# ==============================

# Load the dataset containing customer spending data
df_customer_data = pd.read_csv(r'C:\codebase\FinBytes\customer_data.csv')

# Group data by 'Customer ID' and sum the spending amounts for each customer
customer_spending = df_customer_data.groupby('Customer ID')['Amount'].sum()

# Create a bar chart to visualize total spending per customer
customer_spending.plot(kind='bar')

# Set the title and labels for the chart
plt.title('Customer Spending Patterns')
plt.xlabel('Customer ID')
plt.ylabel('Total Amount Spent')

# Display the chart
plt.show()


# ==============================
# 3. Predicting Loan Default Rates
# ==============================

# Load the dataset containing loan data
df_loan_data = pd.read_csv(r'C:\codebase\FinBytes\loan_data.csv')

# Split the data into features (X) and target variable (y), where 'Default' is the target
X_train, X_test, y_train, y_test = train_test_split(df_loan_data.drop('Default', axis=1), df_loan_data['Default'], test_size=0.3)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Create a DataFrame to compare actual vs predicted values
report = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Display the report
print("\nLoan Default Prediction Report:")
print(report)

