import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# first we set the parameters of our strategy
FAST = 12 # period of fast MA, in case FAST=1 we have the close price
SLOW = 26 # period of slow MA
SGNLN = 9 # period of signal line
DOLLARTC = 0.05 # transaction costs per one buy/sell transaction, is influenced by the number of shares we trade
INITIALAMOUNT = 500 # the initial amount of money we start trading with


data = yf.download(tickers = 'BTC-USD', start='2000-01-01', end='2022-11-01', interval = '1d', group_by = 'column', auto_adjust = True)
#data = yf.download(tickers = 'SPY', start='2000-01-01', end='2022-11-01', interval = '1d', group_by = 'column', auto_adjust = True)


data['Fast'] = data['Close'].ewm(span=FAST, adjust=False).mean()
data['Slow'] = data['Close'].ewm(span=SLOW, adjust=False).mean()
data['MACD'] = data['Fast'] - data['Slow']
#data['MACD'] = data['MACD'] / data['Close']
data['Signal line'] = data['MACD'].ewm(span=SGNLN, adjust=False).mean() 

conditions = [data['MACD']>data['Signal line'], data['MACD']<data['Signal line']] # conditions to check
choices = [1, -1] # the position to substitute
# choices = [1, 0] # alternatively we can consider long-noly postions
defaultchoice = 0 # position if none of conditions is evaluated as True, i.e. in case that fastMA=slowMA or in case of NaN

data['Position'] = np.select(condlist=conditions, choicelist=choices, default=defaultchoice) # calculate the postion based on conditions

# now the question is for what price we buy and for what price we sell, see that the position is calculated based on the close price, so we probably cannot buy for that price
# let's suppose we cannot buy/sell for the close price we use to calculate MA
# if we suppose that we CAN buy/sell for the close price we use to compute MA, comment the following line
data['Position'] = data['Position'].shift(1) # we shift the postion by one day

data.loc[data.index[-1],'Position'] = 0 # at the end we close all positions
data['Trade'] = data['Position'].diff() # we calculate what should we do, i.e. should we sell the long postion and open short postion, or cover short position and buy a long position
data['CF'] = -data['Close'] * data['Trade'] - DOLLARTC * np.abs(data['Trade']) # we calculate profit in currecny for trading one share
data['Profit'] = data['CF'].cumsum() + data['Position'] * data['Close']

# calculate profit of B&H strategy 
data['Profit B&H'] = data['Close'] - data.loc[data.index[0],'Close'] - DOLLARTC 
data.loc[data.index[-1], 'Profit B&H'] = data.loc[data.index[-1], 'Profit B&H'] - DOLLARTC 

print(f"No. of stcoks traded in total: {data['Trade'].abs().sum()}.")
print(f"Final profit: ${data['Profit'][-1].round(2)}.")
print(f"Profit of B&H strategy: ${data['Profit B&H'][-1].round(2)}.")



# visualize 
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(19.20,10.80)) # create the figure with 3 plots/graphs
fig.suptitle('MA trading system') # add the title to the figure
ax1.plot(data[['Close', 'Fast', 'Slow']]) # in the first graph, plot the stock price (close), fast and slow MA
ax2.plot(data[['MACD', 'Signal line']])
ax3.plot(data[['Position', 'Trade']]) # in the second graph, plot suggested position and quantity traded
ax4.plot(data[['Profit', 'Profit B&H']]) 
ax1.set_ylabel('Price & MAs') # set the label of y axis
ax2.set_ylabel('MACD & Signal line') # set the label of y axis
ax3.set_ylabel('Position & trade signal') # set the label of y axis
ax4.set_ylabel('Total profit') # set the label of y axis
ax4.legend(['Strategy','B & H'])
del ax1, ax2, ax3, ax4  # delete unnecessary variables



# ------------------------------------------------------------------------------------
# alernatively we assume that we have a certain ammount of funds with which we trade 
