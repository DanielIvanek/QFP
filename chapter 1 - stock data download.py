import numpy as np
import yfinance as yf  # we import the package and alias it to yf
  
tickers=["MSFT", "AAPL", "TSLA", "PG"]  # we specify the list of tickers we want to obtain
 
data = yf.download(tickers, start="2017-01-01", end="2017-04-30", interval = "1d", group_by="ticker", auto_adjust = True)
  
data = yf.download(tickers, start="2017-01-01", end="2017-04-30", interval = "1d", group_by="column", auto_adjust = True)

prices=data["Close"]
  
# we can calculate percentage changes (simple/discrete returns)
pct_returns = prices.pct_change().dropna()
# or we can calculate the log-returns
log_returns = np.log(prices/prices.shift(1)).dropna()
  
print(log_returns.describe())
log_returns.info()
print(log_returns.head())
 
