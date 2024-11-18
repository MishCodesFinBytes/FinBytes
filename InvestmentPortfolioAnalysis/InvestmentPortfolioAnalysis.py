"""
A Comprehensive Guide to Enhancing Investment Portfolios in Fintech
Sections:
1. Automating Calculations
2. Data Analysis
3. Predictive Modeling
"""

# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Automating Calculations ---
# Calculate the expected return and risk (standard deviation) of a portfolio

# Historical yearly returns for a portfolio (in decimal form)
portfolio_returns = np.array([0.05, 0.08, 0.12, 0.07])  # Example data

# Calculate the expected return
expected_return = portfolio_returns.mean()

# Calculate portfolio risk (standard deviation of returns)
portfolio_risk = portfolio_returns.std()

print(f"Automated Portfolio Metrics:")
print(f" - Expected Return: {expected_return:.2%}")
print(f" - Portfolio Risk (Standard Deviation): {portfolio_risk:.2%}")
print("=" * 50)

# --- 2. Data Analysis ---
# Visualizing historical portfolio returns

# Corresponding years for the historical data
years = [2016, 2017, 2018, 2019]

# Plot the historical returns
plt.figure(figsize=(8, 5))
plt.plot(years, portfolio_returns, marker='o', linestyle='--', color='b', label="Returns")
plt.axhline(y=expected_return, color='r', linestyle='-', label="Expected Return")
plt.title("Historical Portfolio Returns", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Return (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- 3. Predictive Modeling ---
# Forecasting future returns using a simple linear regression model

# Training data: Year (independent variable) and return (dependent variable)
X_train = np.array(years).reshape(-1, 1)  # Reshape for sklearn compatibility
y_train = portfolio_returns

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict returns for future years (2020-2025)
future_years = np.array([2020, 2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)
predicted_returns = model.predict(future_years)

# Evaluate model performance on training data
training_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)

print(f"Predictive Modeling Results:")
print(f" - Training RMSE: {training_rmse:.4f}")
print(f" - Predicted Returns (2020-2025):")
for year, pred in zip(future_years.flatten(), predicted_returns):
    print(f"   {year}: {pred:.2%}")

# Visualize actual vs. predicted returns
plt.figure(figsize=(8, 5))
plt.plot(years, portfolio_returns, marker='o', label="Actual Returns")
plt.plot(future_years.flatten(), predicted_returns, marker='x', linestyle='--', label="Predicted Returns")
plt.title("Portfolio Return Predictions", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Return (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



