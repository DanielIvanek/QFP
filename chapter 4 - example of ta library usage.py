import yfinance as yf
import ta
import matplotlib.pyplot as plt

# Download historical stock data
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Calculate On-Balance Volume (OBV) and Relative Strength Index (RSI)
data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

# Plot Close Price and OBV on the top subplot
ax1.plot(data['Close'], label='AAPL Close Price', color='blue')
ax1.set_ylabel('Close Price', color='blue')
ax1.legend(loc='upper left')

ax1_2 = ax1.twinx()
ax1_2.plot(data['OBV'], label='On-Balance Volume (OBV)', color='orange')
ax1_2.set_ylabel('On-Balance Volume (OBV)', color='orange')
ax1_2.legend(loc='upper right')

# Plot RSI on the bottom subplot
ax2.plot(data['RSI'], label='Relative Strength Index (RSI)', color='green', linestyle='dashed')
ax2.set_xlabel('Date')
ax2.set_ylabel('RSI', color='green')
ax2.legend(loc='upper right')

# Show the plot
plt.title('AAPL Stock Price with On-Balance Volume (OBV) and RSI')
plt.show()