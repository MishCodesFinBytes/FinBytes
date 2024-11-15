# Import necessary libraries
import requests  # For HTTP requests to Alpha Vantage API
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import yfinance as yf  # For retrieving data from Yahoo Finance

### Alpha Vantage API Example ###
# Replace with your own Alpha Vantage API key
api_key = "API_KEY"

# Stock symbol to fetch data for (e.g., Apple Inc.)
stock_symbol = "AAPL"

# URL to access Alpha Vantage's TIME_SERIES_DAILY endpoint
alpha_vantage_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={api_key}"

# Fetch the data from Alpha Vantage
print("Fetching daily stock data from Alpha Vantage...")
response = requests.get(alpha_vantage_url)
data = response.json()

# Extract the daily time series data
daily_data = data.get("Time Series (Daily)", {})

# Convert the daily data into a Pandas DataFrame
# The API returns prices as strings, so they need to be converted to numeric types
df_daily = pd.DataFrame(daily_data).T
df_daily.columns = ["Open", "High", "Low", "Close", "Volume"]
df_daily = df_daily.astype(float)
df_daily.index = pd.to_datetime(df_daily.index)  # Convert index to datetime for better plotting
df_daily.sort_index(inplace=True)  # Ensure data is sorted by date

# Plot the daily close prices
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily["Close"], label="Daily Close Price", color="green", linewidth=1.5)
plt.title("Apple (AAPL) Daily Close Prices from Alpha Vantage", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

### Yahoo Finance API Example ###
# Set the date range for historical data
start_date = "2021-01-01"
end_date = "2021-06-30"

# Fetch stock data for Apple from Yahoo Finance
print("\nFetching stock data from Yahoo Finance...")
aapl = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate the 30-day Moving Average for the adjusted closing price
aapl["30-day MA"] = aapl["Adj Close"].rolling(window=30).mean()

# Plot Adjusted Close prices with the 30-day Moving Average
plt.figure(figsize=(12, 6))
plt.plot(aapl.index, aapl["Adj Close"], label="Adjusted Close", color="blue", linewidth=1.5)
plt.plot(aapl.index, aapl["30-day MA"], label="30-day Moving Average", color="orange", linestyle="--", linewidth=1.5)
plt.title("Apple (AAPL) Stock Price and 30-day Moving Average", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
