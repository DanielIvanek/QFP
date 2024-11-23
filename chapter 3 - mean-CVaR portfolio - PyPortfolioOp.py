# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypfopt

# Make sure to install the required PyPortfolioOpt package if not already installed:
# !pip install PyPortfolioOpt

# Define the in-sample period for historical data analysis.
start_date = '2012-01-01'
end_date = '2017-01-01'

## Download the list of DJIA tickers
## Note: To avoid survivorship bias, the composition of the DJIA index should ideally be taken from the last day of the in-sample period.
#url = 'https://stockmarketmba.com/stocksinthedjia.php'
## Retrieve the table containing the tickers using pandas
#ticker_list = pd.read_html(url)[1]['Symbol'].dropna().tolist()
### alternatively you can download the csv file from https://stockmarketmba.com/stocksinthedjia.php ans save it as 'DJIAlist.csv'
## TICKERLIST = pd.read_csv("DJIAlist.csv")['Symbol'].dropna().tolist() # load the csv into the Pandas Dataframe


url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
ticker_list = pd.read_html(url)[1]['Symbol'].dropna().tolist()

# Download historical stock data using the yfinance library.
# We will only keep the closing prices for our analysis.
prices = yf.download(tickers=ticker_list, start=start_date, end=end_date, interval='1d', group_by='column', auto_adjust=True)["Close"]
# Remove any columns with missing values to maintain data consistency
prices.dropna(axis=1, inplace=True)

# Calculate expected returns and historical returns using PyPortfolioOpt functions
mu = pypfopt.expected_returns.mean_historical_return(prices)
historical_returns = pypfopt.expected_returns.returns_from_prices(prices)

# Initialize DataFrame to stroe the weights
weights = pd.DataFrame(index=prices.columns)

# Portfolio Optimization: Minimum CVaR (Conditional Value at Risk)
beta = 0.95  # Confidence level used for CVaR calculation
# Initialize the Efficient Frontier object for CVaR
ef_cvar = pypfopt.efficient_frontier.EfficientCVaR(expected_returns=mu, returns=historical_returns, weight_bounds=(0, 1))
# Optimize for minimum CVaR
weights_min_cvar = ef_cvar.min_cvar()
# Retrieve and clean the weights for the minimum CVaR portfolio
weights['min CVaR'] = pd.Series(ef_cvar.clean_weights())
# Calculate and store the performance of the minimum CVaR portfolio
ret_min_cvar, cvar_min = ef_cvar.portfolio_performance()

# Portfolio Optimization: Maximum Expected Return
# Reinitialize the Efficient Frontier object
ef_max_ret = pypfopt.efficient_frontier.EfficientCVaR(expected_returns=mu, returns=historical_returns, weight_bounds=(0, 1))
# Optimize for the portfolio that maximizes expected return for the given level of CVaR
ef_max_ret.efficient_risk(target_cvar=1)
# Retrieve and clean the weights for the maximum return portfolio
weights['max ret'] = pd.Series(ef_max_ret.clean_weights())
# Calculate and store the performance of the maximum return portfolio
ret_max, cvar_max = ef_max_ret.portfolio_performance()

# Generate and collect performance data for a range of portfolios
returns = []
cvars = []
n_portfolios = 50  # Number of portfolios to simulate

# Loop through a range of target returns to generate different portfolios
for target_ret in np.linspace(ret_min_cvar, ret_max, n_portfolios):
    # Reinitialize the Efficient Frontier object
    ef = pypfopt.efficient_frontier.EfficientCVaR(expected_returns=mu, returns=historical_returns, weight_bounds=(0, 1))
    # Optimize for the efficient return for each target return
    ef.efficient_return(target_return=target_ret)
    # Retrieve and clean the weights for the current portfolio
    weights['Target return of {:.2f}%'.format(target_ret * 100)] = pd.Series(ef.clean_weights())
    # Calculate and store the performance of the current portfolio
    ret, cvar = ef.portfolio_performance()
    returns.append(ret)
    cvars.append(cvar)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.plot(cvars, returns, label='Efficient Frontier')
plt.title('Efficient Frontier Curve')
plt.xlabel('Conditional Value at Risk (CVaR)')
plt.ylabel('Expected Return')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the weights as a stacked barchart
weights_transposed = weights.drop(columns=['min CVaR', 'max ret']).T
weights_transposed.plot(kind='bar', stacked=True, figsize=(10, 6), legend=False)
plt.title('Portfolio Weights for Different Optimization Strategies')
plt.xlabel('Assets')
plt.ylabel('Weights')
#plt.legend(title='Optimization Strategies', loc='best')
plt.tight_layout()
plt.show()