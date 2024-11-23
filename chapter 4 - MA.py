import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# first we set the parameters of our strategy
FAST = 3 # period of fast MA, in case FAST=1 we have the close price
SLOW = 21 # period of slow MA
DOLLARTC = 0.05 # transaction costs per one buy/sell transaction
INITIALAMOUNT = 500 # the initial amount of money we start trading with

# first we download the data
data = yf.download(tickers = 'SPY', start='2000-01-01', end='2022-11-01', interval = '1d', group_by = 'column', auto_adjust = True)

data['Fast'] = data['Close'].rolling(FAST).mean() # we calculate the fast mooving average
data['Slow'] = data['Close'].rolling(SLOW).mean() # we calculate the slow mooving average

conditions = [data['Fast']>data['Slow'], data['Fast']<data['Slow']] # conditions to check
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


# visualize - only last 125 days
# create the figure with 3 plots/graphs
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(19.20,10.80)) 
fig.suptitle('MA trading system') # add the title to the figure
# in the first graph, plot the stock price (close), fast and slow MA
ax1.plot(data[['Close', 'Fast', 'Slow']]) 
# in the second graph, plot the position and quantity traded (buy/sell)
ax2.plot(data[['Position', 'Trade']]) 
# in the third graph, plot the cumulatative profit
ax3.plot(data[['Profit', 'Profit B&H']]) 
ax1.set_ylabel('Price & MAs') # set the label of y axis
ax2.set_ylabel('Position & trade signal') # set the label of y axis
ax3.set_ylabel('Total profit') # set the label of y axis
ax1.legend(['Close', 'Fast MA', 'Slow MA']) # add legend
ax2.legend(['Position', 'Buy/Sell']) # add legend
ax3.legend(['Strategy','B & H']) # add legend
plt.tight_layout() 


# visualize - only last 125 days
# create the figure with 3 plots/graphs
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(19.20,10.80)) 
fig.suptitle('MA trading system') # add the title to the figure
# in the first graph, plot the stock price (close), fast and slow MA
ax1.plot(data[['Close', 'Fast', 'Slow']][-125:]) 
# in the second graph, plot the position and quantity traded (buy/sell)
ax2.plot(data[['Position', 'Trade']][-125:]) 
# in the third graph, plot the cumulatative profit
ax3.plot(data[['Profit', 'Profit B&H']][-125:]) 
ax1.set_ylabel('Price & MAs') # set the label of y axis
ax2.set_ylabel('Position & trade signal') # set the label of y axis
ax3.set_ylabel('Total profit') # set the label of y axis
ax1.legend(['Close', 'Fast MA', 'Slow MA']) # add legend
ax2.legend(['Position', 'Buy/Sell']) # add legend
ax3.legend(['Strategy','B & H']) # add legend
plt.tight_layout() 




del ax1, ax2, ax3  # delete unnecessary variables