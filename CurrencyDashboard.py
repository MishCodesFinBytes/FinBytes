import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import yfinance as yf
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    # Header for the app
    html.H1("Currency Dashboard"),
    
    # Dropdown for currency pair selection
    html.Div([
        dcc.Dropdown(
            id='currency-pair',  # Dropdown ID for currency selection
            options=[  # List of currency options to choose from
                {'label': 'US Dollar to Euro (USD to EUR)', 'value': 'USDEUR=X'},
                {'label': 'US Dollar to British Pound (USD to GBP)', 'value': 'USDGBP=X'},
                {'label': 'US Dollar to Japanese Yen (USD to JPY)', 'value': 'USDJPY=X'},
                {'label': 'Euro to British Pound (EUR to GBP)', 'value': 'EURGBP=X'},
                {'label': 'Australian Dollar to US Dollar (AUD to USD)', 'value': 'AUDUSD=X'},
                {'label': 'US Dollar to Canadian Dollar (USD to CAD)', 'value': 'USDCAD=X'},
                {'label': 'US Dollar to Swiss Franc (USD to CHF)', 'value': 'USDCHF=X'},
                {'label': 'US Dollar to Australian Dollar (USD to AUD)', 'value': 'USDAUD=X'},
                {'label': 'US Dollar to New Zealand Dollar (USD to NZD)', 'value': 'USDNZD=X'},
                {'label': 'US Dollar to Mexican Peso (USD to MXN)', 'value': 'USDMXN=X'},
                {'label': 'US Dollar to South Korean Won (USD to KRW)', 'value': 'USDKRW=X'},
                {'label': 'US Dollar to Indian Rupee (USD to INR)', 'value': 'USDINR=X'},
                {'label': 'US Dollar to Chinese Yuan (USD to CNY)', 'value': 'USDCNY=X'},
                {'label': 'US Dollar to Brazilian Real (USD to BRL)', 'value': 'USDBRL=X'},
                {'label': 'US Dollar to Singapore Dollar (USD to SGD)', 'value': 'USDSGD=X'}
            ],
            value='USDEUR=X',  # Default value to USD to EUR
            style={'width': '50%'}
        ),
    ], style={'padding': '10px'}),
    
    # Real-time conversion input fields
    html.Div([
        html.Label("Amount to Convert:"),  # Label for input field
        dcc.Input(id='amount', type='number', value=1, style={'width': '10%'})  # Amount input
    ], style={'padding': '10px'}),
    
    # Display the converted amount
    html.Div([
        html.Label("Converted Amount:"),
        html.Div(id='converted-amount', style={'fontSize': 20, 'fontWeight': 'bold'})
    ], style={'padding': '10px'}),
    
    # Historical chart for exchange rates
    html.Div([
        html.H3("Historical Exchange Rates"),
        dcc.Graph(id='historical-graph')  # Graph for historical data
    ], style={'padding': '10px'}),
])

# Callback function to update real-time conversion
@app.callback(
    Output('converted-amount', 'children'),
    [Input('currency-pair', 'value'),  # Get selected currency pair
     Input('amount', 'value')]  # Get amount entered by user
)
def update_conversion(currency_pair, amount):
    data = yf.Ticker(currency_pair)  # Fetch currency pair data using yfinance
    ticker_data = data.history(period="1d")  # Get the latest data for the pair
    conversion_rate = ticker_data['Close'].iloc[0]  # Get the closing price (conversion rate)
    converted_value = amount * conversion_rate  # Calculate the converted value
    return f"{round(converted_value, 2)}"  # Return the converted amount rounded to 2 decimals

# Callback function to update the historical graph
@app.callback(
    Output('historical-graph', 'figure'),
    [Input('currency-pair', 'value')]  # Get selected currency pair
)
def update_historical_graph(currency_pair):
    data = yf.Ticker(currency_pair)  # Fetch currency pair data
    historical_data = data.history(period="1y")  # Get one year of historical data
    
    # Plot the historical exchange rates as a candlestick chart
    figure = {
        'data': [
            go.Candlestick(
                x=historical_data.index,  # X-axis: Dates
                open=historical_data['Open'],  # Open prices
                high=historical_data['High'],  # High prices
                low=historical_data['Low'],  # Low prices
                close=historical_data['Close'],  # Close prices
                name='Exchange Rate'  # Name of the data series
            )
        ],
        'layout': go.Layout(
            title=f"Historical Data for {currency_pair}",  # Title of the chart
            xaxis={'rangeslider': {'visible': False}},  # Hide range slider
            yaxis={'title': 'Exchange Rate (USD)'}  # Label for the Y-axis
        )
    }
    return figure  # Return the plotly figure to be displayed

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


