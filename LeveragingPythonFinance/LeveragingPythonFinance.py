import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt


# Define the portfolio returns
portfolio_returns = pd.Series([0.05, 0.1, 0.08, 0.12, 0.15])

# Calculate the average annual return
avg_return = portfolio_returns.mean()

# Calculate the standard deviation of returns
std_dev = portfolio_returns.std()

# Calculate the risk-free rate of return
risk_free_rate = 0.02

# Calculate the Sharpe Ratio
sharpe_ratio = (avg_return - risk_free_rate) / std_dev

# Print the result
print("The Sharpe Ratio for the portfolio is:", sharpe_ratio)


# Read the stock prices data
stock_prices = pd.read_csv('C:\codebase\FinBytes\stock_prices.csv')

# Plot the stock prices over time
plt.plot(stock_prices['Date'], stock_prices['Price'])
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Prices over Time')
plt.show()

# Convert 'Date' column to datetime format
stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])

# Convert dates to the number of days since the first date in the dataset
stock_prices['Date'] = (stock_prices['Date'] - stock_prices['Date'].min()).dt.days

# Create the feature matrix (use the Date as a feature)
X = stock_prices[['Date']]  # Only using Date as the feature for prediction

# Create the target vector (Price is what we are predicting)
y = stock_prices['Price']

# Fit the linear regression model
model = LinearRegression().fit(X, y)

# Make a prediction for a future date (e.g., July 25, 2025)
future_date = pd.to_datetime('2025-07-25')
minDate = pd.to_datetime(stock_prices['Date'].min())

# Convert the future date to number of days since the minimum date in the dataset
days_since_start = (future_date-minDate).days  

# Predict the stock price for the future date
prediction = model.predict([[days_since_start]])

# Print the predicted stock price
print(f"The predicted stock price for {future_date.strftime('%B %d, %Y')} is: {prediction[0]:.2f}")
