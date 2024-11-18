"""
A Comprehensive Guide to Enhancing Investment Portfolios in Fintech
Sections:
1. Automating Calculations
2. Data Analysis
3. Predictive Modeling
"""

# --- Import necessary libraries ---
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Automating Calculations ---
# Fetch historical stock data for selected tickers
tickers = ['AAPL', 'TSLA', 'GE', 'XOM']
start_date = '2018-01-01'
end_date = '2023-01-01'

# Download adjusted closing prices
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate expected annual return and annualized volatility for each stock
expected_annual_returns = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)

print("Automated Portfolio Metrics:")
for ticker in tickers:
    print(f" - {ticker}:")
    print(f"   - Expected Annual Return: {expected_annual_returns[ticker]:.2%}")
    print(f"   - Annual Volatility (Risk): {annual_volatility[ticker]:.2%}")
print("=" * 50)

# --- 2. Data Analysis ---
# Visualizing historical stock prices
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(data.index, data[ticker], label=ticker)
plt.title("Historical Stock Prices (2018-2023)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Adjusted Close Price (USD)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- 3. Predictive Modeling ---
# Forecasting future returns using a simple linear regression model

# Prepare data for modeling
# We'll use the average of all stock returns as the target variable
returns['Average'] = returns.mean(axis=1)

# Shift the target variable to predict the next day's average return
returns['Target'] = returns['Average'].shift(-1)
returns = returns.dropna()

# Features: daily returns of individual stocks
X = returns[tickers]
# Target: next day's average return
y = returns['Target']

# Split data into training and testing sets
split_index = int(len(returns) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
test_rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Predictive Modeling Results:")
print(f" - Test RMSE: {test_rmse:.4f}")

# Visualize actual vs. predicted returns
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label="Actual Average Return", color='blue')
plt.plot(y_test.index, y_pred, label="Predicted Average Return", color='red', linestyle='--')
plt.title("Actual vs. Predicted Average Returns", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Return", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- 4. Portfolio Performance Analysis ---
# Define an example investment portfolio with allocations to each stock
portfolio_allocations = {'AAPL': 0.4, 'TSLA': 0.3, 'GE': 0.2, 'XOM': 0.1}

# Normalize weights to ensure they sum to 1
weights = np.array(list(portfolio_allocations.values()))

# Filter returns DataFrame to include only the selected tickers
filtered_returns = returns[tickers]

# Calculate portfolio returns using the filtered returns
portfolio_daily_returns = filtered_returns.dot(weights)

# Calculate cumulative portfolio returns
cumulative_returns = (1 + portfolio_daily_returns).cumprod()

# Visualize portfolio performance over time
plt.figure(figsize=(14, 7))
plt.plot(cumulative_returns.index, cumulative_returns, label="Portfolio Performance", color='green')
plt.title("Portfolio Cumulative Performance (2018-2023)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Cumulative Return in USD", fontsize=12)
plt.axhline(y=1, color='black', linestyle='--', label="Initial Investment")
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Report portfolio metrics
final_value = cumulative_returns.iloc[-1]
print(f"Portfolio Performance Analysis:")
print(f" - Final Value of $1 Investment: ${final_value:.2f}")
print(f" - Total Return: {(final_value - 1) * 100:.2f}%")
print("=" * 50)
