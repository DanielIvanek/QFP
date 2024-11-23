import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

# Download historical data for desired ticker symbol
ticker = "SPY"
data = yf.download(ticker, start='2022-10-01', end='2022-11-30', interval = '1d', group_by = 'column', auto_adjust = True)

# Create subplots
fig, axs = plt.subplots(2,2, figsize=(19, 10))

# Plot line chart
mpf.plot(data, type='line', ax=axs[0,0], volume=False, show_nontrading=False)

# Plot bar chart
mpf.plot(data, type='ohlc', ax=axs[0,1], volume=False, show_nontrading=False)

# Plot candlestick chart
mpf.plot(data, type='candle', ax=axs[1,0], volume=False, show_nontrading=False)

# Plot PnF chart
mpf.plot(data, type='pnf', ax=axs[1,1], volume=False, show_nontrading=False,
         pnf_params=dict(box_size=2, reversal=3))

# Display the plot
plt.tight_layout()
plt.show()
