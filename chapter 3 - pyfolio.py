import pyfolio as pf
import yfinance as yf


prices = yf.download(tickers = 'INTC', start='2012-01-01', end='2023-01-01', interval = '1d', group_by = 'column', auto_adjust = True)
returns = prices['Close'].pct_change().dropna() #calculate returns

# calculate the statistics of the wealth path 
print(pf.timeseries.perf_stats(returns))

pf.create_returns_tear_sheet(returns)
#pf.create_simple_tear_sheet(returns)
#pf.create_full_tear_sheet(returns)