# Import necessary libraries
import requests
import json
from bs4 import BeautifulSoup


def get_stock_price_from_alpha_vantage(symbol, api_key):
    """
    Fetch the latest stock price for a given symbol using Alpha Vantage API.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'MSFT' for Microsoft).
        api_key (str): Your Alpha Vantage API key.

    Returns:
        str: Latest stock price as a string, or an error message.
    """
    try:
        # API Endpoint with query parameters
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
        
        # Make HTTP GET request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Extract the latest stock price
        latest_price = data['Global Quote']['05. price']
        return f"Alpha Vantage - Latest Price for {symbol}: {latest_price} USD"
    
    except KeyError:
        return "Error: Could not extract stock price. Check the API response structure."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def get_stock_price_from_yahoo_finance(symbol):
    """
    Fetch the latest stock price for a given symbol using Yahoo Finance.

    Args:
        symbol (str): The stock ticker symbol (e.g., 'MSFT' for Microsoft).

    Returns:
        str: Latest stock price as a string, or an error message.
    """
    try:
        # Yahoo Finance URL for the given ticker
        url = f'https://finance.yahoo.com/quote/{symbol}'
        
        # Make HTTP GET request
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse HTML response using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Dynamically locate the span containing the stock price
        price_tag = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
        
        if not price_tag:
            return "Error: Could not extract stock price. Check the webpage structure."

        # Extract and return the price
        latest_price = price_tag.text
        return f"Yahoo Finance - Latest Price for {symbol}: {latest_price} USD"
    
    except AttributeError:
        return "Error: Could not extract stock price. Check the webpage structure."
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    # Replace with your Alpha Vantage API key
    API_KEY = "YOUR_API_KEY"
    SYMBOL = "MSFT"

    # Fetch stock price using Alpha Vantage API
    print(get_stock_price_from_alpha_vantage(SYMBOL, API_KEY))

    # Fetch stock price using Yahoo Finance
    print(get_stock_price_from_yahoo_finance(SYMBOL))
