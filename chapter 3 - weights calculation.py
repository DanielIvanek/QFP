import yfinance as yf  # we import the package and alias it to yf
import pandas as pd

tickers = ["META",  # Facebook/Meta Platforms, Inc.
           "AAPL",  # Apple
           "AMZN",  # Amazon
           "NFLX",  # Netflix
           "GOOGL"  # Google (Alphabet Inc.)
          ]
prices = yf.download(tickers, start="2013-01-02", end="2013-01-03", interval = "1d", group_by="column", auto_adjust = True,)["Close"]


# defining portfolio by v
v = pd.DataFrame(data=[1, 1, 1, 1, 1], index=tickers, columns=['weight']) # quantitites held
u = v * prices.iloc[-1:].transpose().values  # weights in absolute value (in $)
W = u.sum()  # portoflio value
w = u /W # relative weights (in %)

print(f'Portfolio value(W): {W[0]:.2f}')
print(f'Portfolio quantities(v): {v}')
print(f'Portfolio absolute weights(u): {u}')
print(f'Portfolio relative weights(w): {w}')


# defining portfolio by W and w
W = 1000
w = pd.DataFrame(data=1/5, index=tickers, columns=['weight']) # quantitites held
u = w.multiply(W)  # weights in absolute value (in $)
v = u.div(prices.iloc[-1:].values.transpose())

print(f'Portfolio value(W): {W}')
print(f'Portfolio quantities(v): {v}')
print(f'Portfolio absolute weights(u): {u}')
print(f'Portfolio relative weights(w): {w}')