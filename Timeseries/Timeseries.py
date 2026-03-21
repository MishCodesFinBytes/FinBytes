import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Load financial data
df = pd.read_csv('C:/codebase/FinBytes/financial_data.csv')

# Convert "Date" column to DateTimeIndex and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate daily stock returns using log difference
df['Returns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

# Plotting the daily stock returns
df['Returns'].plot(title="Daily Stock Returns", ylabel="Returns")
plt.show()

# === Part 2: Using numpy ===
# Reset the index for numpy operations
df.reset_index(inplace=True)

# Calculate a 30-day moving average of stock prices using numpy convolve
df['MA'] = np.convolve(df['Close'], np.ones(30)/30, mode='same')

# Plot the 30-day moving average
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['MA'], label="30-Day Moving Average", color="orange")
plt.title("30-Day Moving Average of Stock Prices")
plt.xlabel("Date")
plt.ylabel("Moving Average")
plt.legend()
plt.show()

# === Part 3: Using matplotlib ===
# Plotting the stock prices over time
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Close'], label="Stock Prices")
plt.title("Stock Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()

# === Part 4: Using Prophet ===
# Preparing data for Prophet
prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize Prophet model
model = Prophet()
model.fit(prophet_df)

# Making a future dataframe for prediction (30 days ahead)
future = model.make_future_dataframe(periods=365)
future.tail()

# Forecasting
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# Plotting the forecast
model.plot(forecast)
plt.show()
