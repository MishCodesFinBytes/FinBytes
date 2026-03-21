# Import necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Define the list of stocks to analyze
stocks = ['AAPL', 'AMZN', 'MSFT']

# Set plot style for better visualization
sns.set(style="whitegrid")

# Fetch financial data (Adjusted Close prices)
# This will fetch the adjusted close prices from 2021-01-01 to 2021-05-31
financial_data = yf.download(stocks, start='2021-01-01', end='2021-05-31')['Adj Close']

# Display the first few rows of the financial data for inspection
print("Financial Data (Adjusted Close):")
print(financial_data.head())

# --- P/E Ratio Calculation ---
# P/E Ratio = Price / Earnings (Let’s assume we have earnings data as 'Earnings' for each stock)

# For demonstration, let's use hypothetical earnings data for each stock (replace with actual data in practice)
earnings_data = {
    'AAPL': 5.11,  # Hypothetical earnings per share for Apple
    'AMZN': 41.83,  # Hypothetical earnings per share for Amazon
    'MSFT': 8.05    # Hypothetical earnings per share for Microsoft
}

# --- Apply P/E Ratio Calculation ---
# Ensure we divide each stock’s adjusted closing price by its respective earnings per share (EPS)
def calculate_pe_ratio(stock_name, adjusted_close_price):
    earnings_per_share = earnings_data.get(stock_name)
    if earnings_per_share is not None:
        return adjusted_close_price / earnings_per_share
    else:
        return None  # Return None if no earnings data is available

# Apply P/E ratio calculation to the adjusted close prices
for stock in stocks:
    financial_data[stock + ' P/E Ratio'] = financial_data[stock].apply(lambda x: calculate_pe_ratio(stock, x))

# Display the data with P/E Ratio
print("\nFinancial Data with P/E Ratios:")
print(financial_data.tail())

# --- Sort by P/E Ratio ---
# Calculate the average P/E ratio for each stock and sort them in ascending order (low P/E is better)
pe_ratios = {stock: financial_data[stock + ' P/E Ratio'].mean() for stock in stocks}
sorted_pe_data = pd.Series(pe_ratios).sort_values(ascending=True)

print("\nStocks Sorted by P/E Ratio (Lowest to Highest):")
print(sorted_pe_data)

# --- Dividend Yield Calculation ---
# Fetch dividend data for stocks
# Note: Adjusting for the latest API methods since 'get_dividends' is deprecated in newer versions of yfinance
dividends = {stock: yf.Ticker(stock).dividends['2021-01-01':'2021-05-31'] for stock in stocks}

# Create a new DataFrame to hold the dividend yield for each stock
dividend_yields = {}

for stock in stocks:
    # Calculate the dividend yield = Dividends / Adjusted Close Price
    stock_dividends = dividends[stock]
    adj_close = financial_data[stock]
    
    # Calculate dividend yield (annualized)
    dividend_yield = stock_dividends.sum() / adj_close.mean()  # Summing up dividends in the period
    dividend_yields[stock] = dividend_yield

# Convert dictionary to a pandas DataFrame
dividend_yield_df = pd.DataFrame.from_dict(dividend_yields, orient='index', columns=['Dividend Yield'])

# Sort by Dividend Yield (highest yield first)
sorted_dividends = dividend_yield_df.sort_values(by='Dividend Yield', ascending=False)
print("\nDividend Yields Sorted (Highest to Lowest):")
print(sorted_dividends)

# --- Visualization: P/E Ratio and Dividend Yield Comparison ---
plt.figure(figsize=(12, 6))

# Plot P/E Ratios
plt.subplot(1, 2, 1)
sorted_pe_data.plot(kind='bar', color='lightblue')
plt.title('P/E Ratio Comparison (2021)')
plt.ylabel('P/E Ratio')
plt.xlabel('Stock')

# Plot Dividend Yields
plt.subplot(1, 2, 2)
sorted_dividends.plot(kind='bar', color='lightgreen')
plt.title('Dividend Yield Comparison (2021)')
plt.ylabel('Dividend Yield')
plt.xlabel('Stock')

# Show the plots
plt.tight_layout()
plt.show()

# Conclusion
# Based on the analysis above, investors can look at the P/E ratio for stock valuation and dividend yield for income generation potential.
