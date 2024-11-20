import requests
import pandas as pd
import matplotlib.pyplot as plt

# CoinGecko API endpoint
api_url = "https://api.coingecko.com/api/v3/simple/price" # needs modification  - needs registration and API key

# User's cryptocurrency portfolio
portfolio = {'btc': 2, 'eth': 5, 'ada': 1000, 'sol': 50}

# Fetch cryptocurrency data
def fetch_crypto_data(crypto_symbols):
    """
    Fetch cryptocurrency data for a given list of symbols.
    :param crypto_symbols: List of cryptocurrency symbols (e.g., ['btc', 'eth'])
    :return: DataFrame containing crypto data
    """
    try:
        params = {
            "ids": ",".join(crypto_symbols),
            "vs_currencies": "usd"
        }
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame.from_dict(data, orient="index").reset_index()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Fetch data for portfolio cryptocurrencies
portfolio_symbols = list(portfolio.keys())
crypto_data = fetch_crypto_data(portfolio_symbols)

print(crypto_data)

if not crypto_data.empty:
    # Rename columns for better readability
    crypto_data.columns = ['Cryptocurrency', 'Price (USD)']

    # Add a 'Quantity' column to the DataFrame
    crypto_data['Quantity'] = crypto_data['Cryptocurrency'].map(portfolio)

    # Calculate portfolio value for each cryptocurrency
    crypto_data['Value (USD)'] = crypto_data['Price (USD)'] * crypto_data['Quantity']

    # Display the portfolio summary
    print("\nPortfolio Summary:")
    print(crypto_data[['Cryptocurrency', 'Quantity', 'Price (USD)', 'Value (USD)']])

    # Plot a bar chart of portfolio value
    plt.figure(figsize=(10, 6))
    plt.bar(crypto_data['Cryptocurrency'], crypto_data['Value (USD)'], color='skyblue')
    plt.xlabel('Cryptocurrency')
    plt.ylabel('Portfolio Value (USD)')
    plt.title('Crypto Portfolio Performance')
    plt.tight_layout()
    plt.show()

    # Calculate and display portfolio total value
    total_value = crypto_data['Value (USD)'].sum()
    print(f"\nTotal Portfolio Value: ${total_value:.2f}")
else:
    print("No data available to display.")
