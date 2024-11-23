import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

DOLLARTC = 0.05 # transaction costs per one buy/sell transaction, is influenced by the number of shares we trade

data = yf.download(tickers = 'INTC', start='2000-01-01', end='2022-11-01', interval = '1d', group_by = 'column', auto_adjust = True)

def mastrategy(data,fast,slow):
    data['Fast'] = data['Close'].rolling(fast).mean()
    data['Slow'] = data['Close'].rolling(slow).mean()

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
    
    return data.loc[data.index[-1], 'Profit']

profits = np.zeros(shape=(50,200)) 
for f in range(1,50):  # we will try all values in interval <1,50), ie. including 1 and excluding 50
    for s in range(1,200): # we will try all values in interval <1,200), ie. including 1 and excluding 200
        profits[f,s] = mastrategy(data=data,fast=f,slow=s)
        print(f"Strategy: fast={f}, slow={s} calculated.")
        
fig, ax = plt.subplots(figsize=(19.20,10.80))
im=ax.imshow(profits[1:,1:], cmap=mpl.colormaps['RdYlGn'])
ax.set_xlabel('period of slow MA') # set the label of x axis
ax.set_ylabel('period of fast MA') # set the label of y axis
fig.colorbar(im, location='bottom')
plt.tight_layout()
plt.savefig("hapter 5 - MAcrossover - combinations.pdf", dpi=500)
plt.savefig("chapter 5 - MAcrossover - combinations.png", dpi=500)
plt.show()