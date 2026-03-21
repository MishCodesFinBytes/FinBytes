# Install required libraries if not already installed
# pip install yfinance matplotlib ta pandas

# Import necessary libraries
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import yfinance as yf  # For fetching stock data
import ta  # For technical analysis calculations

# ------------------------
# Example 1: Apple Stock - 10-day SMA
# ------------------------

# Fetch Apple stock data
df_aapl = yf.download('AAPL', start='2020-01-01', end='2020-12-31')

# Calculate 10-day Simple Moving Average (SMA)
sma_10 = df_aapl['Close'].rolling(window=10).mean()

# Plot Apple stock prices with 10-day SMA
plt.figure(figsize=(10, 5))
plt.plot(df_aapl['Close'], label='AAPL Closing Price', linewidth=1)
plt.plot(sma_10, label='10-day SMA', linestyle='--')
plt.title('Apple Stock - 10-day SMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# ------------------------
# Example 2: Microsoft Stock - 20-day EMA
# ------------------------

# Fetch Microsoft stock data
df_msft = yf.download('MSFT', start='2020-01-01', end='2020-12-31')

# Calculate 20-day Exponential Moving Average (EMA)
ema_20 = df_msft['Close'].ewm(span=20, adjust=False).mean()

# Plot Microsoft stock prices with 20-day EMA
plt.figure(figsize=(10, 5))
plt.plot(df_msft['Close'], label='MSFT Closing Price', linewidth=1)
plt.plot(ema_20, label='20-day EMA', linestyle='--')
plt.title('Microsoft Stock - 20-day EMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# ------------------------
# Example 3: Google Stock - 5-day WMA
# ------------------------

# Fetch Google stock data
df_googl = yf.download('GOOGL', start='2020-01-01', end='2020-12-31')

# Function to calculate Weighted Moving Average (WMA)
def calculate_wma(data, window):
    weights = range(1, window + 1)
    return data.rolling(window).apply(lambda prices: sum(prices * weights) / sum(weights), raw=True)

# Calculate 5-day WMA
wma_5 = calculate_wma(df_googl['Close'], 5)

# Plot Google stock prices with 5-day WMA
plt.figure(figsize=(10, 5))
plt.plot(df_googl['Close'], label='GOOGL Closing Price', linewidth=1)
plt.plot(wma_5, label='5-day WMA', linestyle='--')
plt.title('Google Stock - 5-day WMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# ------------------------
# Example 4: Tesla Stock - 14-day RSI
# ------------------------

# Fetch Tesla stock data
df_tsla = yf.download('TSLA', start='2020-01-01', end='2020-12-31')

# Ensure 'Close' is treated as a Series
if isinstance(df_tsla['Close'], pd.DataFrame):
    # Convert to Series if it's a DataFrame
    close_prices = df_tsla['Close'].squeeze()
else:
    close_prices = df_tsla['Close']

# Confirm the shape of 'Close'
print(f"Shape of 'Close': {close_prices.shape}")

# Calculate 14-day Relative Strength Index (RSI)
rsi_14 = ta.momentum.rsi(close_prices, window=14)

# Plot Tesla stock prices with 14-day RSI
plt.figure(figsize=(10, 5))

# Subplot 1: Tesla Closing Prices
plt.subplot(2, 1, 1)
plt.plot(close_prices, label='TSLA Closing Price', linewidth=1)
plt.title('Tesla Stock - Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Subplot 2: RSI
plt.subplot(2, 1, 2)
plt.plot(rsi_14, label='14-day RSI', linestyle='--')
plt.axhline(70, color='red', linestyle='--', linewidth=0.5, label='Overbought')
plt.axhline(30, color='green', linestyle='--', linewidth=0.5, label='Oversold')
plt.title('Tesla Stock - 14-day RSI')
plt.xlabel('Date')
plt.ylabel('RSI Value')
plt.legend()

plt.tight_layout()
plt.show()


# ------------------------
# Example 5: Amazon Stock - MACD
# ------------------------

# Fetch Amazon stock data
df_amzn = yf.download('AMZN', start='2020-01-01', end='2020-12-31')

# Ensure 'Close' is treated as a Series
if isinstance(df_amzn['Close'], pd.DataFrame):
    # Convert to Series if it's a DataFrame
    close_prices = df_amzn['Close'].squeeze()
else:
    close_prices = df_amzn['Close']

# Confirm the shape of 'Close'
print(f"Shape of 'Close': {close_prices.shape}")

# Calculate Moving Average Convergence Divergence (MACD)
macd_line = ta.trend.macd(close_prices, window_slow=26, window_fast=12)
signal_line = ta.trend.macd_signal(close_prices, window_slow=26, window_fast=12, window_sign=9)

# Plot Amazon stock prices with MACD and Signal line
plt.figure(figsize=(10, 5))

# Subplot 1: Amazon Closing Prices
plt.subplot(2, 1, 1)
plt.plot(close_prices, label='AMZN Closing Price', linewidth=1)
plt.title('Amazon Stock - Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Subplot 2: MACD and Signal Line
plt.subplot(2, 1, 2)
plt.plot(macd_line, label='MACD Line', linestyle='--')
plt.plot(signal_line, label='Signal Line', linestyle=':')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5, label='Zero Line')
plt.title('Amazon Stock - MACD')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------
# Example 6: Facebook Stock - Stochastic Oscillator
# ------------------------



# Fetch Meta Platforms stock data from Yahoo Finance
df_meta = yf.download('META', start='2020-01-01', end='2020-12-31')

# Extracting the 'High', 'Low', and 'Close' columns as 1D series
high = df_meta['High'].squeeze()  # Ensures 1-dimensional
low = df_meta['Low'].squeeze()    # Ensures 1-dimensional
close = df_meta['Close'].squeeze()  # Ensures 1-dimensional

# Calculate Stochastic Oscillator
stochastic_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
stochastic_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

# Plot Meta Platforms stock prices with Stochastic Oscillator
plt.figure(figsize=(10, 5))
plt.plot(close, label='META Closing Price', linewidth=1)
plt.plot(stochastic_k, label='%K Line', linestyle='--')
plt.plot(stochastic_d, label='%D Line', linestyle=':')
plt.title('Meta Platforms Stock - Stochastic Oscillator')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
