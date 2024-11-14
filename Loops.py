# Loan Payment Calculation

# Initialize the loan amount, interest rate, and loan duration in years
loan_amount = 100000
interest_rate = 0.05
years = 5

# Calculate the monthly interest rate
monthly_interest = interest_rate / 12

# Loop through each month in the loan period to calculate the monthly payment
for i in range(years * 12):
    # Calculate the monthly payment using the loan amortization formula
    monthly_payment = (loan_amount * monthly_interest) / (1 - (1 + monthly_interest) ** -(years * 12))
    
    # Print the monthly payment for each month
    print("Month", i + 1, "Monthly Payment:", round(monthly_payment, 2))


# Stock Price Monitoring

import yfinance as yf  # Import the Yahoo Finance library for stock price data

# Set the stock symbol and target price
stock_symbol = "AAPL"
target_price = 150

# Continuously monitor the stock price until it reaches the target
while True:
    # Retrieve the current stock price
    stock_price = yf.Ticker(stock_symbol).info['currentPrice']
    
    # Check if the stock price has reached or exceeded the target price
    if stock_price >= target_price:
        # Print a message if the target price is reached
        print("Stock price for", stock_symbol, "has reached or exceeded the target price of", target_price, "at", stock_price)
        break
