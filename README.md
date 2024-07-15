# Beta-and-Capital-Asset-Pricing-Model-using-Python

This repository contains a Python script to analyze the performance of various stocks compared to the S&P 500 index using historical data from Yahoo Finance.

## Overview

The script performs the following tasks:
1. Fetches historical stock data for specified symbols over a defined date range.
2. Cleans and processes the data to remove missing values.
3. Calculates average prices, daily returns, and beta values for each stock.
4. Normalizes stock prices for comparison.
5. Generates interactive plots to visualize the stock performance.
6. Calculates and compares expected returns using the Capital Asset Pricing Model (CAPM).

## Parameters

- `symbols`: Dictionary containing stock symbols and their corresponding tickers.
- `start_date`: Start date for fetching historical data (YYYY-MM-DD).
- `end_date`: End date for fetching historical data (YYYY-MM-DD).
- `rf`: Risk-free rate (used for CAPM calculation).
- `duration`: Total simulation time (s).
- `dt`: Time step (s).

## Key Functions

1. **Normalize Prices**
   - Normalizes the stock prices based on the initial price.
   - Function: `normalize(data)`

2. **Interactive Plot**
   - Generates interactive line plots for the stock prices.
   - Function: `interactive_plot(data, title)`

3. **Calculate Daily Returns**
   - Calculates the daily returns for each stock in percentage.
   - Function: `daily_return(data)`

4. **Calculate Beta and Alpha**
   - Calculates beta and alpha values for the stocks relative to the S&P 500 index.
   - Uses these values to calculate expected returns using CAPM.

## Output

The script generates the following outputs:
1. Descriptive statistics and information of the stock data.
2. Average prices of each stock.
3. Interactive plots for the stock prices and normalized prices.
4. Scatter plots and regression lines for the daily returns of stocks compared to the S&P 500 index.
5. Beta, alpha, expected return (CAPM), and actual return for each stock.

## Usage

1. Clone the repository to your local machine.
2. Install the necessary Python libraries using the following command:
   ```sh
   pip install pandas seaborn plotly yfinance matplotlib numpy
