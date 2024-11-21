# Mean Reversion Portfolio Optimization Script
# Author: [Your Name]
# Date: [Today's Date]

import pandas as pd
import numpy as np
import requests
import cvxpy as cp
from datetime import datetime, timedelta
import time
import urllib.parse

# Replace with your actual Polygon.io API key
API_KEY = 'GNthmWT9qYGm57QwnIJ_orim_uN5mbc0'

# Function to fetch the list of S&P 500 tickers
def get_sp500_tickers():
    url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    try:
        sp500_df = pd.read_csv(url)
        tickers = sp500_df['Symbol'].tolist()
        # Do not replace dots in tickers; use them as is
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

# Ensure we have enough tickers
tickers = get_sp500_tickers()

# Function to fetch historical closing prices
def fetch_close_prices(ticker, start_date, end_date):
    encoded_ticker = urllib.parse.quote_plus(ticker)
    url = f"https://api.polygon.io/v2/aggs/ticker/{encoded_ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 5000,
        'apiKey': API_KEY
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data for {ticker}: {response.status_code} {response.text}")
            return pd.Series(dtype='float64')
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
        print(f"Exception fetching data for {ticker}: {e}")
        return pd.Series(dtype='float64')

# Function to fetch the latest price
def fetch_latest_price(ticker):
    encoded_ticker = urllib.parse.quote_plus(ticker)
    url = f"https://api.polygon.io/v2/last/trade/{encoded_ticker}"
    params = {'apiKey': API_KEY}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching latest price for {ticker}: {response.status_code} {response.text}")
            return None
        data = response.json()
        if 'results' in data and 'p' in data['results']:
            price = data['results']['p']
            return price
        else:
            print(f"Could not fetch latest price for {ticker}")
            return None
    except Exception as e:
        print(f"Exception fetching latest price for {ticker}: {e}")
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
    valid_tickers = []

    for ticker in tickers:
        print(f"Processing {ticker}")
        prices = fetch_close_prices(ticker, start_date, end_date)
        latest_price = fetch_latest_price(ticker)
        if latest_price is None or latest_price < 5 or prices.empty or len(prices) <= 20:
            print(f"Skipping {ticker} due to insufficient data or price below $5")
            continue
        # Ensure the ticker is valid and consistent
        valid_tickers.append(ticker)
        price_data[ticker] = prices
        latest_prices[ticker] = latest_price
        # Respect API rate limits
        time.sleep(0.25)

    # Ensure we have enough stocks
    if len(valid_tickers) < 82:
        raise ValueError("Not enough stocks with valid data to select 41 long and 41 short positions.")

    # Create a DataFrame of historical prices
    prices_df = pd.DataFrame(price_data).dropna(axis=0, how='any')
    returns_df = prices_df.pct_change().dropna()

    # Calculate expected returns and covariance matrix
    expected_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    # Calculate Z-scores for mean reversion signal
    z_scores = {}
    for ticker in valid_tickers:
        prices = price_data[ticker]
        z = calculate_z_score(prices)
        latest_z = z.iloc[-1]
        z_scores[ticker] = latest_z

    z_scores_df = pd.DataFrame.from_dict(z_scores, orient='index', columns=['z_score'])
    z_scores_df['price'] = z_scores_df.index.map(latest_prices)
    z_scores_df.dropna(inplace=True)

    # Sort stocks based on Z-score
    z_scores_df.sort_values('z_score', inplace=True)

    # Ensure we have at least 41 long and 41 short positions
    num_positions = 41
    if len(z_scores_df) < 2 * num_positions:
        raise ValueError(f"Not enough stocks to select {num_positions} long and {num_positions} short positions.")

    long_tickers = z_scores_df.head(num_positions).index.tolist()
    short_tickers = z_scores_df.tail(num_positions).index.tolist()
    selected_tickers = long_tickers + short_tickers

    # Validate selected_tickers against expected_returns
    missing_tickers = [ticker for ticker in selected_tickers if ticker not in expected_returns.index]
    if missing_tickers:
        print(f"Tickers missing from expected_returns: {missing_tickers}")
        # Remove missing tickers from selected_tickers
        selected_tickers = [ticker for ticker in selected_tickers if ticker in expected_returns.index]
        # Update long_tickers and short_tickers
        long_tickers = [ticker for ticker in long_tickers if ticker in selected_tickers]
        short_tickers = [ticker for ticker in short_tickers if ticker in selected_tickers]
        if len(long_tickers) < 41 or len(short_tickers) < 41:
            raise ValueError("Not enough valid tickers to proceed with 41 long and 41 short positions.")

    mu = expected_returns[selected_tickers]
    Sigma = cov_matrix.loc[selected_tickers, selected_tickers]

    # Define variables for optimization
    w = cp.Variable(len(selected_tickers))
    risk_aversion = 1  # Adjust risk aversion coefficient as needed

    # Desired invested capital and maximum weight per stock
    desired_invested_capital = 0.955  # 95.5% invested capital
    max_weight_per_stock = 0.025      # 2.5% maximum weight per stock
    min_weight_per_stock = 0.005      # 0.5% minimum weight per stock to avoid zero weights after rounding

    sum_long_weights = desired_invested_capital / 2
    sum_short_weights = -desired_invested_capital / 2

    # Set up the optimization problem
    # Objective: Maximize expected return minus risk penalty
    objective = cp.Maximize(mu.values @ w - risk_aversion * cp.quad_form(w, Sigma.values))

    # Constraints
    constraints = [
        cp.sum(w[0:num_positions]) == sum_long_weights,    # Sum of long weights equals 47.75%
        cp.sum(w[num_positions:]) == sum_short_weights,    # Sum of short weights equals -47.75%
        w[0:num_positions] >= min_weight_per_stock,        # Long positions weights are at least 0.5%
        w[num_positions:] <= -min_weight_per_stock,        # Short positions weights are at least -0.5%
        w >= -max_weight_per_stock,                        # No position less than -2.5% of total portfolio
        w <= max_weight_per_stock,                         # No position greater than 2.5% of total portfolio
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
    except Exception as e:
        print(f"Optimization failed: {e}")
        return

    if w.value is None:
        print("Optimization did not find a solution.")
        return

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

    # Ensure no position exceeds 2.5% of total portfolio value
    max_position_value = max_weight_per_stock * total_portfolio_value
    if portfolio['position_value'].abs().max() > max_position_value:
        print("Adjusting positions to not exceed maximum position value per stock...")
        portfolio['position_value'] = portfolio['position_value'].apply(
            lambda x: np.sign(x) * min(abs(x), max_position_value))
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
        portfolio['shares'] = portfolio['position_value'] / portfolio['price']

    # Ensure minimum position size to avoid zero shares after rounding
    min_position_value = min_weight_per_stock * total_portfolio_value
    portfolio['position_value'] = portfolio['position_value'].apply(
        lambda x: np.sign(x) * max(abs(x), min_position_value))
    portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
    portfolio['shares'] = portfolio['position_value'] / portfolio['price']

    # Round shares to at least 1 to avoid zero weights
    portfolio['shares'] = portfolio['shares'].apply(lambda x: np.ceil(abs(x)) * np.sign(x))

    # Recalculate position values and weights based on integer shares
    portfolio['position_value'] = portfolio['shares'] * portfolio['price']
    portfolio['weight'] = portfolio['position_value'] / total_portfolio_value

    # Recalculate invested capital and cash
    invested_capital = portfolio['position_value'].abs().sum()
    cash = total_portfolio_value - invested_capital
    cash_ratio = cash / total_portfolio_value

    # Adjust positions to achieve 4.5% cash
    if abs(cash_ratio - 0.045) > 0.001:
        print(f"Cash ratio is {cash_ratio:.4f}, adjusting positions to achieve 4.5% cash...")
        adjustment_factor = (total_portfolio_value * (1 - 0.045)) / invested_capital
        portfolio['shares'] = portfolio['shares'] * adjustment_factor
        portfolio['shares'] = portfolio['shares'].apply(lambda x: np.ceil(abs(x)) * np.sign(x))
        portfolio['position_value'] = portfolio['shares'] * portfolio['price']
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
        # Recalculate invested capital and cash
        invested_capital = portfolio['position_value'].abs().sum()
        cash = total_portfolio_value - invested_capital
        cash_ratio = cash / total_portfolio_value
        print(f"Adjusted cash ratio: {cash_ratio:.4f}")

    # Ensure no position exceeds maximum weight after adjustments
    portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
    if portfolio['weight'].abs().max() > max_weight_per_stock:
        print("Adjusting positions to not exceed maximum weight per stock after rounding shares...")
        scaling_factor = max_weight_per_stock / portfolio['weight'].abs().max()
        portfolio['shares'] *= scaling_factor
        portfolio['shares'] = portfolio['shares'].apply(lambda x: np.ceil(abs(x)) * np.sign(x))
        portfolio['position_value'] = portfolio['shares'] * portfolio['price']
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value

    # Final recalculations
    invested_capital = portfolio['position_value'].abs().sum()
    cash = total_portfolio_value - invested_capital
    cash_ratio = cash / total_portfolio_value

    # Check dollar neutrality ratio
    total_long_value = portfolio[portfolio['position'] == 'long']['position_value'].sum()
    total_short_value = -portfolio[portfolio['position'] == 'short']['position_value'].sum()
    dollar_neutrality_ratio = total_long_value / total_short_value

    print(f"Dollar Neutrality Ratio: {dollar_neutrality_ratio:.2f}")

    if not 0.9 <= dollar_neutrality_ratio <= 1.1:
        print("Adjusting positions to meet dollar neutrality ratio constraints...")
        scaling_factor = (total_short_value / total_long_value)
        portfolio.loc[portfolio['position'] == 'long', 'shares'] *= scaling_factor
        portfolio.loc[portfolio['position'] == 'long', 'shares'] = portfolio.loc[portfolio['position'] == 'long', 'shares'].apply(lambda x: np.ceil(abs(x)) * np.sign(x))
        portfolio['position_value'] = portfolio['shares'] * portfolio['price']
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value
        # Recalculate totals
        total_long_value = portfolio[portfolio['position'] == 'long']['position_value'].sum()
        total_short_value = -portfolio[portfolio['position'] == 'short']['position_value'].sum()
        dollar_neutrality_ratio = total_long_value / total_short_value
        print(f"Adjusted Dollar Neutrality Ratio: {dollar_neutrality_ratio:.2f}")

    # Ensure we have at least 41 long and 41 short positions with nonzero weights
    num_long_nonzero = (portfolio.loc[portfolio['position'] == 'long', 'shares'] != 0).sum()
    num_short_nonzero = (portfolio.loc[portfolio['position'] == 'short', 'shares'] != 0).sum()

    if num_long_nonzero < num_positions or num_short_nonzero < num_positions:
        print(f"Adjusting to ensure at least {num_positions} long and {num_positions} short positions with nonzero weights...")
        # For positions with zero shares, set shares to 1
        portfolio.loc[(portfolio['position'] == 'long') & (portfolio['shares'] == 0), 'shares'] = 1
        portfolio.loc[(portfolio['position'] == 'short') & (portfolio['shares'] == 0), 'shares'] = -1
        portfolio['position_value'] = portfolio['shares'] * portfolio['price']
        portfolio['weight'] = portfolio['position_value'] / total_portfolio_value

    # Final recalculations after ensuring nonzero positions
    invested_capital = portfolio['position_value'].abs().sum()
    cash = total_portfolio_value - invested_capital
    cash_ratio = cash / total_portfolio_value

    # Save the portfolio to a CSV file
    portfolio = portfolio[['ticker', 'z_score', 'expected_return', 'position', 'price', 'shares', 'position_value', 'weight']]
    portfolio.to_csv('mean_reversion_portfolio.csv', index=False)
    print("Portfolio has been saved to 'mean_reversion_portfolio.csv'.")

    # Display the portfolio DataFrame
    print(portfolio)
    return portfolio

if __name__ == "__main__":
    final_portfolio = main()