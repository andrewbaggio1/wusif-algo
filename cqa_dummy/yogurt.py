# Mean Reversion Portfolio Optimization Script
# Author: [Your Name]
# Date: [Today's Date]

import pandas as pd
import numpy as np
import requests
import cvxpy as cp
from datetime import datetime, timedelta
import time

# Replace with your actual Polygon.io API key
API_KEY = 'GNthmWT9qYGm57QwnIJ_orim_uN5mbc0'

# Function to fetch the list of S&P 500 tickers
def get_sp500_tickers():
    url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    try:
        sp500_df = pd.read_csv(url)
        tickers = sp500_df['Symbol'].tolist()
        # Replace any dots in tickers (e.g., BRK.B to BRK-B) to match Polygon's format
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

# Ensure we have enough tickers
tickers = get_sp500_tickers()

# Function to fetch historical closing prices
def fetch_close_prices(ticker, start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            return df['c']  # Closing prices
        else:
            print(f"No data for {ticker}")
            return pd.Series(dtype='float64')
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.Series(dtype='float64')

# Function to fetch the latest price
def fetch_latest_price(ticker):
    url = f"https://api.polygon.io/v2/last/trade/{ticker}"
    params = {'apiKey': API_KEY}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'results' in data and 'p' in data['results']:
            price = data['results']['p']
            return price
        else:
            print(f"Could not fetch latest price for {ticker}")
            return None
    except Exception as e:
        print(f"Error fetching latest price for {ticker}: {e}")
        return None

# Function to calculate Z-score
def calculate_z_score(series, window=20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z_score = (series - mean) / std
    return z_score

# Main script
def main():
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    price_data = {}
    latest_prices = {}

    for ticker in tickers:
        print(f"Processing {ticker}")
        prices = fetch_close_prices(ticker, start_date, end_date)
        latest_price = fetch_latest_price(ticker)
        if latest_price is None or latest_price < 5 or prices.empty or len(prices) <= 20:
            print(f"Skipping {ticker} due to insufficient data or price below $5")
            continue
        price_data[ticker] = prices
        latest_prices[ticker] = latest_price
        # Respect API rate limits
        time.sleep(0.25)

    # Ensure we have enough stocks
    if len(price_data) < 82:
        raise ValueError("Not enough stocks with valid data to select long and short positions.")

    # Create a DataFrame of historical prices
    prices_df = pd.DataFrame(price_data).dropna(axis=1)
    returns_df = prices_df.pct_change().dropna()

    # Calculate expected returns and covariance matrix
    expected_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    # Calculate Z-scores for mean reversion signal
    z_scores = {}
    for ticker, prices in price_data.items():
        z = calculate_z_score(prices)
        latest_z = z.iloc[-1]
        z_scores[ticker] = latest_z

    z_scores_df = pd.DataFrame.from_dict(z_scores, orient='index', columns=['z_score'])
    z_scores_df['price'] = z_scores_df.index.map(latest_prices)
    z_scores_df.dropna(inplace=True)

    # Sort stocks based on Z-score
    z_scores_df.sort_values('z_score', inplace=True)

    # Select stocks for long and short positions
    num_positions = 41
    long_tickers = z_scores_df.head(num_positions).index.tolist()
    short_tickers = z_scores_df.tail(num_positions).index.tolist()
    selected_tickers = long_tickers + short_tickers

    # Subset expected returns and covariance matrix
    mu = expected_returns[selected_tickers]
    Sigma = cov_matrix.loc[selected_tickers, selected_tickers]

    # Define variables for optimization
    w = cp.Variable(len(selected_tickers))
    risk_aversion = 1  # Adjust risk aversion coefficient as needed

    # Set up the optimization problem
    # Objective: Maximize expected return minus risk penalty
    objective = cp.Maximize(mu.values @ w - risk_aversion * cp.quad_form(w, Sigma.values))

    # Constraints
    constraints = [
        cp.sum(w[:num_positions]) == 0.5,  # Sum of long weights equals 0.5 (50% of total portfolio)
        cp.sum(w[num_positions:]) == -0.5,  # Sum of short weights equals -0.5
        w[:num_positions] >= 0,  # Long positions weights are positive
        w[num_positions:] <= 0,  # Short positions weights are negative
        w >= -0.05,  # No position less than -5% of total portfolio
        w <= 0.05,   # No position greater than 5% of total portfolio
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Get the optimized weights
    optimized_weights = w.value

    # Create the portfolio DataFrame
    portfolio = pd.DataFrame({
        'ticker': selected_tickers,
        'weight': optimized_weights,
        'expected_return': mu.values,
        'price': [latest_prices[ticker] for ticker in selected_tickers],
        'z_score': [z_scores[ticker] for ticker in selected_tickers]
    })

    # Calculate shares and position values
    total_portfolio_value = 2000000
    portfolio['position_value'] = portfolio['weight'] * total_portfolio_value
    portfolio['shares'] = portfolio['position_value'] / portfolio['price']
    portfolio['position'] = portfolio['weight'].apply(lambda x: 'long' if x > 0 else 'short')

    # Ensure minimum number of stocks
    num_longs = len(portfolio[portfolio['position'] == 'long'])
    num_shorts = len(portfolio[portfolio['position'] == 'short'])
    if num_longs < 41 or num_shorts < 41:
        raise ValueError("Portfolio does not meet the minimum stock requirement for long and short positions.")

    # Check dollar neutrality ratio
    total_long_value = portfolio[portfolio['position'] == 'long']['position_value'].sum()
    total_short_value = -portfolio[portfolio['position'] == 'short']['position_value'].sum()
    dollar_neutrality_ratio = total_long_value / total_short_value

    print(f"Dollar Neutrality Ratio: {dollar_neutrality_ratio:.2f}")

    if not 0.9 <= dollar_neutrality_ratio <= 1.1:
        print("Adjusting positions to meet dollar neutrality ratio constraints...")
        # Adjust positions proportionally
        scaling_factor = (total_long_value + total_short_value) / (2 * total_long_value)
        portfolio.loc[portfolio['position'] == 'long', 'weight'] *= scaling_factor
        portfolio.loc[portfolio['position'] == 'long', 'position_value'] = portfolio['weight'] * total_portfolio_value
        portfolio.loc[portfolio['position'] == 'long', 'shares'] = portfolio['position_value'] / portfolio['price']
        # Recalculate totals
        total_long_value = portfolio[portfolio['position'] == 'long']['position_value'].sum()
        total_short_value = -portfolio[portfolio['position'] == 'short']['position_value'].sum()
        dollar_neutrality_ratio = total_long_value / total_short_value
        print(f"Adjusted Dollar Neutrality Ratio: {dollar_neutrality_ratio:.2f}")

    # Ensure cash held is not more than 5% of total portfolio value
    invested_capital = portfolio['position_value'].abs().sum()
    cash = total_portfolio_value - invested_capital
    cash_ratio = cash / total_portfolio_value
    if cash_ratio > 0.05:
        print(f"Cash ratio is {cash_ratio:.2f}, which exceeds the limit. Adjusting positions...")
        adjustment_factor = total_portfolio_value * 0.95 / invested_capital
        portfolio['position_value'] *= adjustment_factor
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
        portfolio['shares'] = portfolio['position_value'] / portfolio['price']
        invested_capital = portfolio['position_value'].abs().sum()
        cash = total_portfolio_value - invested_capital
        cash_ratio = cash / total_portfolio_value
        print(f"Adjusted cash ratio: {cash_ratio:.2f}")

    # Final portfolio adjustments to meet constraints
    # Ensure no position exceeds 5% of total portfolio value
    max_position_value = 0.05 * total_portfolio_value
    if portfolio['position_value'].abs().max() > max_position_value:
        print("Adjusting positions to not exceed maximum position value per stock...")
        portfolio['position_value'] = portfolio['position_value'].apply(
            lambda x: np.sign(x) * min(abs(x), max_position_value))
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
        portfolio['shares'] = portfolio['position_value'] / portfolio['price']

    # Return the final portfolio DataFrame
    portfolio = portfolio[['ticker', 'z_score', 'expected_return', 'position', 'price', 'shares', 'position_value', 'weight']]

    # Optionally, you can output the portfolio to a CSV file
    portfolio.to_csv('optimized_mean_reversion_portfolio.csv', index=False)
    print("Portfolio has been saved to 'optimized_mean_reversion_portfolio.csv'.")

    # Display the portfolio DataFrame
    print(portfolio)
    return portfolio

if __name__ == "__main__":
    final_portfolio = main()
