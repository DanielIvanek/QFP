import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download data
tickers = ["META",  # Facebook/Meta Platforms, Inc.
           "AAPL",  # Apple
           "AMZN",  # Amazon
           "NFLX",  # Netflix
           "GOOGL"  # Google (Alphabet Inc.)
          ]



#tickers = ["TSLA", # Tesla
#           "INTC"  # Intel
#          ]



data = yf.download(tickers=tickers, start='2013-01-01', end='2022-12-31', interval='1d', group_by='column', auto_adjust=True)
prices = data["Close"].dropna()


# Initial weights and portfolio value
v = pd.DataFrame(index=prices.columns, columns=['weight'], data=1)
#v = pd.DataFrame(index=prices.columns, columns=['weight'], data=[1,5])
initial_portfolio_value = prices.iloc[0].dot(v).iloc[0]
initial_w = prices.iloc[0].multiply(v['weight']) / initial_portfolio_value

# Calcualation of returns time series
simple_returns = prices.pct_change().fillna(0)
log_returns = np.log(prices / prices.shift(1)).fillna(0)

cumulative_log_returns = log_returns.cumsum()
cumulative_gross_simple_returns = np.exp(cumulative_log_returns)
# alternatively we calculate from simple returns 
cumulative_gross_simple_returns = (simple_returns + 1).cumprod()

print(initial_w)
print(initial_portfolio_value)


# 1. Wealth path via simple approach (rebalancing)
reb_weights = pd.DataFrame(index=simple_returns.index, columns=initial_w.index)
reb_weights[:] = initial_w.values
# vector multiplication returns matrix multiplied by vector of weights
reb_portfolio_return = simple_returns.dot(initial_w)
# alternatively element by element multiplication of returns matrix by weights matrix
reb_portfolio_return2 = simple_returns.multiply(reb_weights).sum(axis=1)
reb_wealth_path_relative = (1+reb_portfolio_return).cumprod()
reb_wealth_path_absolute = reb_wealth_path_relative * initial_portfolio_value


# 2. Wealth path evolution
fir_wealth_path_absolute = prices.dot(v)
fir_wealth_path_relative = prices.dot(v) / initial_portfolio_value


# 3. Cumulative returns
sec_wealth_path_relative  = cumulative_gross_simple_returns.dot(initial_w) 
sec_wealth_path_absolute  = cumulative_gross_simple_returns.dot(initial_w) * initial_portfolio_value.sum()


# 4. New weights matrix
# Correct weights calculation
weights = pd.DataFrame(index=simple_returns.index, columns=initial_w.index)
weights.iloc[0] = initial_w # Set the initial weights for the first date
# Iterate over each date and recalculate the weights
for i in range(1, len(simple_returns)):
    weights.iloc[i] = weights.iloc[i-1] * (1 + simple_returns.iloc[i-1]) # Calculate the new weights based on returns
    weights.iloc[i] /= weights.iloc[i].sum() # Normalize the weights so they sum up to 1
thi_portfolio_return = simple_returns.multiply(weights).sum(axis=1)
thi_wealth_path_relative = (1+thi_portfolio_return).cumprod()
thi_wealth_path_absolute = thi_wealth_path_relative * initial_portfolio_value


# Plot the resuls

# Plotting reb_weights and weights
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
reb_weights.plot.area(ax=axes[0], stacked=True, title="Rebalanced Weights Over Time")
axes[0].set_ylabel("Weights")
axes[0].legend(loc="upper right")
weights.plot.area(ax=axes[1], stacked=True, title="Weights Over Time (Without Rebalancing)")
axes[1].set_ylabel("Weights")
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.show()


# Plotting relative and absolute wealth paths
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# Plot for relative wealth paths
axes[0].plot(reb_wealth_path_relative, label="Rebalanced")
axes[0].plot(fir_wealth_path_relative, label="Quantities")
axes[0].plot(sec_wealth_path_relative, label="Cumulative Returns")
axes[0].plot(thi_wealth_path_relative, label="Dynamic Weights")
axes[0].set_title("Relative Wealth Paths Over Time")
axes[0].set_ylabel("Relative Wealth")
axes[0].legend(loc="upper left")

# Plot for absolute wealth paths
axes[1].plot(reb_wealth_path_absolute, label="Rebalanced")
axes[1].plot(fir_wealth_path_absolute, label="Quantities")
axes[1].plot(sec_wealth_path_absolute, label="Cumulative Returns")
axes[1].plot(thi_wealth_path_absolute, label="Dynamic Weights")
axes[1].set_title("Absolute Wealth Paths Over Time")
axes[1].set_ylabel("Absolute Wealth (in USD)")
axes[1].legend(loc="upper left")
plt.tight_layout()
plt.show()



