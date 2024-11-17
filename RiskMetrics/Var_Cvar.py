import numpy as np
import pandas as pd
from scipy.stats import norm

# Load historical returns data from CSV
try:
    returns = pd.read_csv(r'C:\codebase\FinBytes\historical_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file path provided does not exist. Please check the file path.")

# Ensure the DataFrame contains numeric data
returns = returns.select_dtypes(include=[np.number])

# Debugging: Check the structure of the DataFrame
print("Data preview:")
print(returns.head())
print(returns.info())

# Define portfolio weights (adjust as needed)
weights = np.array([0.4, 0.6])  # Example weights

# Ensure weights and returns align
if returns.shape[1] != len(weights):
    raise ValueError(
        f"Mismatch: The returns DataFrame has {returns.shape[1]} columns, but weights array has {len(weights)} elements!"
    )

# === Method 1: Historical VaR Calculation ===

# Calculate portfolio returns
portfolio_returns = returns.dot(weights)

# Sort portfolio returns
sorted_returns = np.sort(portfolio_returns)

# VaR level (95% confidence level)
VaR_level = 0.95

# Calculate VaR as the (1 - VaR_level) percentile
VaR_percentile = (1 - VaR_level) * 100
VaR = np.percentile(sorted_returns, VaR_percentile)

# Convert VaR to monetary value
portfolio_value = 1_000_000  # Portfolio value in USD
VaR_value = portfolio_value * VaR

print(f"Historical VaR: {VaR_value:.2f} USD")

# === Method 2: Parametric VaR and CVaR Calculation ===

# Calculate mean and standard deviation of portfolio returns
portfolio_mean = np.dot(weights, returns.mean())
portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))

# Find z-score for the given VaR level
z_score = norm.ppf(1 - VaR_level)

# Calculate VaR using the z-score
VaR_parametric = portfolio_mean + z_score * portfolio_std
VaR_parametric_value = portfolio_value * VaR_parametric

# Calculate CVaR
CVaR = VaR_parametric_value * (1 - VaR_level)

print(f"Parametric VaR: {VaR_parametric_value:.2f} USD")
print(f"Parametric CVaR: {CVaR:.2f} USD")
