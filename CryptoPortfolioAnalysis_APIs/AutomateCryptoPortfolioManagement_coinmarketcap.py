import requests
import pandas as pd
import matplotlib.pyplot as plt

# Cryptocurrency API endpoint
api_url = 'https://api.coinmarketcap.com/v1/ticker/'

# User's cryptocurrency portfolio
portfolio = {'BTC': 2, 'ETH': 5, 'ADA': 1000, 'SOL': 50}

# Function to fetch cryptocurrency data
def fetch_crypto_data(crypto_symbols):
    """
    Fetch cryptocurrency data for a given list of symbols.
    :param crypto_symbols: List of cryptocurrency symbols (e.g., ['BTC', 'ETH'])
    :return: DataFrame containing crypto data
    """
    crypto_data = []
    for symbol in crypto_symbols:
        endpoint = f"{api_url}{symbol.lower()}/"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()[0]
            crypto_data.append(data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")
    return pd.DataFrame(crypto_data)

# Fetch data for portfolio cryptocurrencies
portfolio_symbols = list(portfolio.keys())
crypto_data = fetch_crypto_data(portfolio_symbols)

# Add a 'Quantity' column to the DataFrame
crypto_data['Quantity'] = crypto_data['symbol'].map(portfolio)

# Convert 'price_usd' to float and calculate portfolio value
crypto_data['price_usd'] = crypto_data['price_usd'].astype(float)
crypto_data['Value (USD)'] = crypto_data['price_usd'] * crypto_data['Quantity']

# Display the portfolio summary
print("\nPortfolio Summary:")
print(crypto_data[['name', 'Quantity', 'price_usd', 'Value (USD)']])

# Plot a bar chart of portfolio value
plt.figure(figsize=(10, 6))
plt.bar(crypto_data['name'], crypto_data['Value (USD)'], color='skyblue')
plt.xlabel('Cryptocurrency')
plt.ylabel('Portfolio Value (USD)')
plt.title('Crypto Portfolio Performance')
plt.tight_layout()
plt.show()

# Calculate and display portfolio total value
total_value = crypto_data['Value (USD)'].sum()
print(f"\nTotal Portfolio Value: ${total_value:.2f}")
