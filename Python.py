#!/usr/bin/env python
# coding: utf-8

# In[17]:


# !pip install yfinance
import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd


# In[19]:


import yfinance as yf


# In[96]:


# Define the stock symbols
symbols = {
    "NFLX": "NFLX",
    "BA": "BA",
    "T": "T",
    "MGM": "MGM",
    "TSLA": "TSLA",
    "sp500": "^GSPC"
}


# In[98]:


# Define the date range
start_date = "2013-01-01"
end_date = "2023-01-01"


# In[100]:


# Fetch the data
data = {}
for symbol, ticker in symbols.items():
    data[symbol] = yf.download(ticker, start=start_date, end=end_date)["Close"]


# In[102]:


# Create a DataFrame
df = pd.DataFrame(data)

# Reset the index to have 'Date' as a column
df.reset_index(inplace=True)


# In[104]:


# Print the DataFrame
df


# In[106]:


# Remove missing values
df.dropna(inplace=True)


# In[108]:


# Describe the data
df.describe()


# In[110]:


# Info of the data
df.info()


# In[112]:


# Average price of each stock
for symbol in symbols:
    avg_price = df[symbol].mean()
    print(f"Average price of {symbol}: {avg_price:.2f}")


# In[114]:


# Function to normalize the prices based on the initial price
def normalize(data):
    x = data.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x

normalize(df)


# In[116]:


# Function to plot interactive plot
def interactive_plot(data, title):
    fig = px.line(title = title)
    for i in data.columns[1:]:
        fig.add_scatter(x = data['Date'], y = data[i], name = i)
    fig.show()


# In[118]:


# Plot interactive chart
interactive_plot(df, 'Prices')

# Plot normalized interactive chart
interactive_plot(normalize(df),'Normalized Prices')


# In[183]:


# Function to calculate the daily returns in percentage


def daily_return(data):
    data_daily_returns = data.copy()
    
    for col in data.columns[1:]:
        data_daily_returns[col] = (data[col] / data[col].shift(1) - 1) * 100
        data_daily_returns.loc[0, col] = 0  # Setting the first day's return to 0
    
    return data_daily_returns
            


# In[185]:


daily_return(df)


# In[121]:


# Select any stock, let's say Apple 
stocks_daily_returns = daily_return(df)


# In[122]:


stocks_daily_returns['NFLX']


# In[128]:


stocks_daily_returns['sp500']


# In[130]:


# plot a scatter plot between the selected stock and the S&P500 (Market)
stocks_daily_returns.plot(kind = 'scatter', x = 'sp500', y = 'NFLX', color = 'black')


# In[132]:


# Fit a polynomial between Netflix and sp500

beta, alpha = np.polyfit(stocks_daily_returns['sp500'], stocks_daily_returns['NFLX'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('NFLX', beta, alpha))  


# In[134]:


# Now let's plot the scatter plot and the straight line on one plot
stocks_daily_returns.plot(kind = 'scatter', x = 'sp500', y = 'NFLX', color = 'black')

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
plt.plot(stocks_daily_returns['sp500'], beta * stocks_daily_returns['sp500'] + alpha, '-', color = 'r')


# In[150]:


# Calculate daily returns for each stock
returns = df[symbols.keys()].pct_change().dropna()

# Calculate beta for each stock relative to S&P500 (^GSPC)
results = {}
for symbol in symbols.keys():
    if symbol != 'sp500':  # Skip calculating beta for the S&P500 itself
        beta, _ = np.polyfit(returns['sp500'], returns[symbol], 1)
        results[symbol] = {
            'beta': beta,
            'alpha': None  # You can calculate alpha if needed
        }

# Print results
for symbol, data in results.items():
    print(f"Stock: {symbol}, Beta: {data['beta']:.4f}")


# In[154]:


# Now doing the same for all the stocks

for symbol, stock in symbols.items():
    if stock == "^GSPC":
        continue
    
    # Calculate beta and alpha
    beta1, alpha = np.polyfit(stocks_daily_returns['sp500'], stocks_daily_returns[stock], 1)
    print('Beta for {} stock is = {} and alpha is = {}'.format(stock, beta1, alpha))
    
    # Plot the scatter plot and the regression line
    fig = px.scatter(stocks_daily_returns, x='sp500', y=stock, trendline='ols', trendline_color_override='red',
                 labels={'sp500': 'S&P 500 Returns', stock: f'{stock} Returns'},
                 title=f'Scatter plot of {stock} vs S&P 500')
    fig.update_layout(title_text=f'Scatter plot of {stock} vs S&P 500', xaxis_title='S&P 500 Returns', yaxis_title=f'{stock} Returns')
    fig.show()


# In[189]:


results = {}
for symbol in symbols.keys():
    if symbol != 'sp500':
        beta, alpha = np.polyfit(stocks_daily_returns['sp500'], stocks_daily_returns[symbol], 1)
        results[symbol] = {
            'Beta': beta,
            'Alpha': alpha
        }
        # Calculate expected return using CAPM
        rf = 0.02  # Assuming a risk-free rate of 2%
        rm = stocks_daily_returns['sp500'].mean() * 252 / 100  # Annualized return of S&P 500
        ER = rf + (beta * (rm - rf))
        results[symbol]['Expected Return (CAPM)'] = ER

        # Actual return for the stock
        actual_return = stocks_daily_returns[symbol].mean() * 252 / 100

        # Compare expected and actual returns
        results[symbol]['Actual Return'] = actual_return

# Print results
for symbol, result in results.items():
    print(f"\nStock: {symbol}")
    print(f"Beta: {result['Beta']:.2f}")
    print(f"Alpha: {result['Alpha']:.2f}")
    print(f"Expected Return (CAPM): {result['Expected Return (CAPM)']:.2%}")
    print(f"Actual Return: {result['Actual Return']:.2%}")

