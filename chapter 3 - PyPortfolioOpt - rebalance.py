import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypfopt

def calc_MDD(prices):
    # the function calculates and returns the Calmar ratio for the price evolution of the stock/portfolio given as the argument
    # the function expect that there is only one column in the variable prices 
    df = pd.DataFrame(prices.copy()) # make a copy of input data
    if (df.shape[1]!=1): # check the input data - whether there is only one column
        raise TypeError("Wrong format of the input data") # if there is not only one column, raise an error and stop the program
    df.rename(columns={df.columns[0]: "prices" }, inplace = True) # rename the first to column to "price" no matter what the name of the column is
    mdd = (-(df['prices']/df['prices'].cummax() - 1)).max() # calculate MDD in one line
    return mdd

def calc_Calmar(prices, rfr=0):
    # the function calculates and returns the Calmar ratio for the price evolution of the stock/portfolio given as the argument
    # the function expect that there is only one column in the variable prices 
    # rfr is the value of risk-free rate to consider (p.a.)
    df = pd.DataFrame(prices.copy()) # make a copy of input data
    if (df.shape[1]!=1): # check the input data - whether there is only one column
        raise TypeError("Wrong format of the input data") # if there is not only one column, raise an error and stop the program
    df.rename(columns={df.columns[0]: "prices" }, inplace = True) # rename the first to column to "price" no matter what the name of the column is
    df['returns'] = df["prices"].pct_change().dropna() # calculate simple returns
    df['returns cumulative gross'] = (1 + df['returns']).cumprod() # calculate the cummulative returns 
    mdd = (-(df['returns cumulative gross']/df['returns cumulative gross'].cummax() - 1)).max() # calculate MDD in one line
    cagr = df['returns cumulative gross'].iloc[-1]**(252/df.shape[0])-1 # calculate Cumulative Annual Growth Rate (CAGR) - the (geometrical) mean annual return
    calmar = (cagr - rfr) / mdd # calculate calmar ratio
    return calmar


def calc_Sharpe(prices, rfr=0):
    # the function calculates and returns the Calmar ratio for the price evolution of the stock/portfolio given as the argument
    # the function expect that there is only one column in the variable prices 
    # rfr is the value of risk-free rate to consider (p.a.)
    df = pd.DataFrame(prices.copy()) # make a copy of input data
    if (df.shape[1]!=1): # check the input data - whether there is only one column
        raise TypeError("Wrong format of the input data") # if there is not only one column, raise an error and stop the program
    df.rename(columns={df.columns[0]: "prices" }, inplace = True) # rename the first to column to "price" no matter what the name of the column is
    df['returns'] = df["prices"].pct_change().dropna() # calculate simple returns
    mr = df['returns'].mean() * 252 # very simple annualizaiton, please refer to CAGR calculation
    std = df['returns'].std() * np.sqrt(252) # very simple annualizaiton,
    sharpe = (mr - rfr) / std # calculate calmar ratio
    return sharpe

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tlSP500 = pd.read_html(url)[0]['Symbol'].tolist()

data = yf.download(tickers = tlSP500, start='2012-01-01', end='2023-01-01', interval = '1d', group_by = 'column', auto_adjust = True)
data = data["Close"].dropna(axis=1, inplace=False) # we drop the symbols with missing data

weights = pd.DataFrame(columns = data.columns)

for year in range(2017, 2022 + 1):
    inprices = data[f'{year-5}-01-01':f'{year}-01-01']
    outprices = data[f'{year}-01-01':f'{year}-12-31']
    outreturns = outprices.pct_change().fillna(0)
    # calculate expected returns and covariance matrix
    mu = pypfopt.expected_returns.mean_historical_return(inprices)
    Q = pypfopt.risk_models.risk_matrix(inprices)
    Q = pypfopt.risk_models.fix_nonpositive_semidefinite(Q)

    # create Efficient Frontier object
    #ef = pypfopt.efficient_frontier.EfficientFrontier(mu, Q)
    ef = pypfopt.efficient_frontier.EfficientCVaR(mu, returns=inprices.pct_change().dropna(), beta=0.95) 

    # calculate weights and store them as w for given year
    w = pd.DataFrame(index=outprices.index, columns=outprices.columns)
    w.iloc[0] = ef.min_cvar() # Set the initial weights for the first date
    # Iterate over each date and recalculate the weights
    for i in range(1, len(outreturns)):
        w.iloc[i] = w.iloc[i-1] * (1 + outreturns.iloc[i-1]) # Calculate the new weights based on returns
        w.iloc[i] /= w.iloc[i].sum() # Normalize the weights so they sum up to 1
    weights = pd.concat([weights, w], axis=0)

weights = weights.astype(float)  
plt.figure(figsize=(10, 6))
plt.stackplot(weights.index, weights.T.values)
plt.ylabel("Weights")
plt.show()
 
outprices = data['2017-01-01':'2023-01-01']
outreturns = outprices.pct_change().fillna(0)

portfolio_return = outreturns.multiply(weights).sum(axis=1)
wealth = (1+portfolio_return).cumprod()

# plot the wealth path evolutions
plt.figure(figsize=(10, 6))
plt.plot(wealth)
plt.title('Wealth paths') # add the title to the figure
plt.ylabel('Portfolio Value') # set the label of y-axis
plt.xlabel('Time') # set the label of x-axis
plt.grid(True)
plt.show()

print(f"Final wealth: ",  wealth.iloc[-1])
print(f"Maximum drawdow: ",  calc_MDD(wealth))
print(f"Sharpe ratio: ",  calc_Sharpe(wealth))
print(f"Calmar ratio: ",  calc_Calmar(wealth))
    
    
"""
for col in wealth.columns:
    print(f"Final wealth of {col}: ",  wealth[col].iloc[-1])

for col in wealth.columns:
    print(f"Maximum drawdow of {col}: ",  calc_MDD(wealth[col]))

for col in wealth.columns:
    print(f"Sharpe ratio of {col}: ",  calc_Sharpe(wealth[col]))

for col in wealth.columns:
    print(f"Calmar ratio of {col}: ",  calc_Calmar(wealth[col]))
"""