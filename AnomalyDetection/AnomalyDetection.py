# ============================
# 1. Statistical Methods
# ============================

# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats

# Load stock prices data from CSV file
df = pd.read_csv(r'C:\codebase\FinBytes\stock_prices_anomaly.csv')

# Ensure the 'Price' column is numeric and handle non-numeric values gracefully
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Check if there are any NaN (missing) values in the 'Price' column
if df['Price'].isnull().any():
    print("Warning: There are missing or invalid entries in the 'Price' column. These will be treated as NaN.")

# Calculate Z-score for each data point in the 'Price' column
df['Z_score'] = np.abs(stats.zscore(df['Price']))

# Set a threshold for detecting anomalies (Z-score > 3 is often considered an outlier)
threshold = 3

# Flag data points with Z-score above the threshold as anomalies
df['Anomaly'] = np.where(df['Z_score'] > threshold, 'Yes', 'No')

# Display flagged anomalies
anomalies_statistical = df[df['Anomaly'] == 'Yes']
print("Anomalies detected using Statistical Methods:")
print(anomalies_statistical)


# ============================
# 2. Machine Learning Algorithms
# ============================

# Import necessary libraries for machine learning
from sklearn.ensemble import IsolationForest

# Load credit card transactions data
df = pd.read_csv(r'C:\codebase\FinBytes\credit_card_transactions.csv')

# Fit an Isolation Forest model to detect anomalies
model = IsolationForest().fit(df[['Amount']])

# Predict outliers using the model
df['Anomaly'] = model.predict(df[['Amount']])

# Display flagged anomalies (Isolation Forest returns -1 for anomalies)
anomalies_ml = df[df['Anomaly'] == -1]
print("\nAnomalies detected using Machine Learning (Isolation Forest):")
print(anomalies_ml)


# ============================
# 3. Time Series Analysis
# ============================


# Import necessary libraries for time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load stock prices data from CSV
df = pd.read_csv(r'C:\codebase\FinBytes\stock_prices_anomaly_TS.csv')

# Ensure the 'Price' column is numeric and handle non-numeric values
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with missing values in the 'Price' column
df = df.dropna(subset=['Price'])

# Perform seasonal decomposition of the time series
# The 'period' parameter controls the cycle length (e.g., 30 for monthly data, use 'period=365' for daily data over a year)
result = seasonal_decompose(df['Price'], model='additive', period=30)

# Extract and store the residuals from the decomposition
df['Residuals'] = result.resid

# Check the residuals to understand their distribution
print("Residuals Summary:")
print(df['Residuals'].describe())

# Perform Augmented Dickey-Fuller (ADF) test to check for stationarity
# ADF test checks if the residuals are stationary (non-random)
adf_result = adfuller(df['Residuals'].dropna())  # Drop NaN values before ADF test

# Extract p-value from ADF test result
p_value = adf_result[1]

# Display ADF test result
print(f"ADF test p-value: {p_value}")

# Set a threshold for p-value (commonly 0.05 for a 95% confidence level)
threshold = 0.05

# Flag residuals with p-value below the threshold as anomalies
if p_value < threshold:
    print(f"The residuals are stationary, p-value: {p_value}")
else:
    print(f"The residuals are non-stationary, p-value: {p_value}")

# Display anomalies (if any)
# In case residuals are flagged as stationary and anomalous, print them
df['Anomaly'] = np.where(df['Residuals'].isna(), 'No', 'Yes')  # Just a placeholder if needed
print("\nAnomalies detected using Time Series Analysis:")
print(df[df['Anomaly'] == 'Yes'])
