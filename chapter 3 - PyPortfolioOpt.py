import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypfopt

# see https://pyportfolioopt.readthedocs.io/

# in-sample period from 2012-01-01 to 2017-01-01
# out-of-sample period from 2017-01-01 to 2023-01-01
 
# Just a note: in this example, our analysis suffers from the so-called survivorship bias due to the fact that we take the composition of the index (and thus our database) from the out-of-sample period
# A better approach would be to take the composition of the DJIA index on the last day of the in-sample period. This would make the analysis survivorship-bias-free.
 
# first donwload the list of tickers in DJIA, see Code 2 1
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tlSP500 = pd.read_html(url)[0]['Symbol'].tolist()
 
## alternatively download the csv file from https://stockmarketmba.com/stocksinthedjia.php and save it as 'DJIAlist.csv'
# tclist = pd.read_csv("DJIAlist.csv")['Symbol'].dropna().tolist() # load the csv into the Pandas Dataframe
 
# Now we download the data using yfinance package and keep only the Close prices (already adjusted)
data = yf.download(tickers = tlSP500, start='2012-01-01', end='2023-01-01', interval = '1d', group_by = 'column', auto_adjust = True)
data = data["Close"].dropna(axis=1, inplace=False) # we drop the symbols with missing data
# we split the data into in-sample and out-of-sample parts, see above
inprices = data[:'2017-01-01'] # the in-sample prices, on these we calculate the weights
outprices =  data['2017-01-01':] # the  out-of-sample prices, on these we check the performance
 
# define a function to return a vector of 1/n weights
def port_naive(prices):
    n = prices.shape[1] # obtain the number of the columns (assets in prices DataFrame)
    weights = [1/n for _ in range(n)]
    return weights
 
# create weights dataframe to stroe the weights of strategies, assign the weights of naive strategy
weights = pd.DataFrame(index=inprices.columns, columns=['naive'], data=port_naive(inprices)) 
 
# calculate the expected returns and covariance matrix, 
# see also other possibilities on how to improve the estimates
mu = pypfopt.expected_returns.mean_historical_return(inprices) # to improve the forecast see also:
#mu = pypfopt.expected_returns.ema_historical_return() 
#mu = pypfopt.expected_returns.capm_return()
 
Q = pypfopt.risk_models.risk_matrix(inprices) # to improve the forecast see the 'method' parameter or pypfopt.risk_models-CovarianceShrinkage(data)
Q = pypfopt.risk_models.fix_nonpositive_semidefinite(Q) # Check the covariance matrix and if not positive semidefinite, fix it.
 
# calculate the weights of the maximum Sharpe ratio portfolio
ef = pypfopt.efficient_frontier.EfficientFrontier(mu, Q) # create the Efficient Frontier object
weights['max Sharpe'] = pd.Series(ef.max_sharpe()) # add the vector of weights to DataFrame
 
# calculate the weights of the minimum variance portfolio
ef = pypfopt.efficient_frontier.EfficientFrontier(mu, Q) # create new Efficient Frontier object
weights['min variance'] = pd.Series(ef.min_volatility())  # add the vector of weights of the minimum variance portfolio to DataFrame 
 
# calculate the minimum-CVaR portfolio (for a confidence level of 95%)
ef = pypfopt.efficient_frontier.EfficientCVaR(mu, returns=inprices.pct_change().dropna(), beta=0.95) # create the Efficient Frontier object
weights['min CVaR'] = pd.Series(ef.min_cvar())  # # add the vector of weights of minimum-CVaR portofolio
 
# the function ef.clean_weights() can be used to get the weights „rounded“ (to getr id of the small weights
#weights['min CVaR'] = pd.Series(ef.clean_weights())
 
# plot the figure with the weights distribution
weights.transpose().plot.bar(stacked=True, legend=False)
plt.title('Weights') # add the title to the figure
plt.ylabel('Weight') # set the label of y axis
#plt.xlabel('Portfolio') # set the label of y axis
# the legend is missing as 29 stocks make the figure very unclear
fig = plt.gcf()
fig.set_size_inches(10, 3)  # Adjust the size of the current figure
plt.show()


#------------------------------------------------------------------------------------------------------------------------------------

# calculate the wealth path evolutions utilizing gross cumulative returns as shown in Code 4 6
wealth = pd.DataFrame(columns=['naive', 'max Sharpe', 'min variance', 'min CVaR'])
wealth['naive'] = ((outprices.pct_change().fillna(0) + 1).cumprod()).dot(weights['naive']) # use the function dot() of the DataFrame with weights in series type
wealth['max Sharpe'] = ((outprices.pct_change().fillna(0) + 1).cumprod()).dot(weights['max Sharpe']) # use the function dot() of the DataFrame with weights in series type
wealth['min variance'] = ((outprices.pct_change().fillna(0) + 1).cumprod()).dot(weights['min variance']) # use the function dot() of the DataFrame with weights in series type
wealth['min CVaR'] = ((outprices.pct_change().fillna(0) + 1).cumprod()).dot(weights['min CVaR']) # use the function dot() of the DataFrame with weights in series type

# plot the wealth path evolutions
plt.figure(figsize=(10, 6))
plt.plot(wealth)
plt.title('Wealth paths') # add the title to the figure
plt.ylabel('Portfolio Value') # set the label of y axis
plt.xlabel('Time') # set the label of y axis
plt.legend(wealth.columns)
plt.show()


def calc_MDD(prices):
    # the function calculates and returns the MDD
    # the function expect that there is only one column in the DataFrame prices or prices are pandas Series
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
    # the function calculates and returns the Sharpe ratio for the price evolution of the stock/portfolio given as the argument
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

for col in wealth.columns:
    print(f"Final wealth {col}: ",  wealth[col].iloc[-1])

for col in wealth.columns:
    print(f"Maximum drawdow of {col}: ",  calc_MDD(wealth[col]))

for col in wealth.columns:
    print(f"Sharpe ratio of {col}: ",  calc_Sharpe(wealth[col]))

for col in wealth.columns:
    print(f"Calmar ratio of {col}: ",  calc_Calmar(wealth[col]))

# see https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficient-semivariance to create efficient semivariance portfolios


# see also other portfolio models: https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficient-semivariance 
# https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html#efficientcdar
# https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html


# Things to do:
# - calculate the performance ratios and compare the wealth path evolutions (use e.g. the function calc_Calmar() from previous file to calculate the Calmar ratio)
# - calculate also other metrices, such as MDD, etc.
# - do figures in plotly.express



# Possible topics for the project:
# - get the dataset of the stock prices and compare different models to each other, chose more than one criterion (e.g. Sharpe ratio, MDD, Calmar ratio, CAGT ....)




"""

groreturns = outprices.pct_change() + 1 # calculate the gross returns as percentage changes + 1
cumreturns = groreturns.cumprod() - 1 # calculate the cumulative returns
pocreturns = cumreturns.dot(weight).to_frame() # calculate the portfolio cummulative returns
wealth = pocreturns + 1 # calculate the wealth path of the portfolio (i.e. the evolution of the value of the portfolio in each time point)
porreturns = wealth.pct_change().dropna() # calculate the portfolio returns
# we do not consider transaction costs - acctualy, as we buy and hold, there would occur transaction costs only two times: at the beginning when we buy the portoflio and at the end when we sell the portfolio

groreturns = outprices.pct_change().fillna(0) + 1 # calculate the gross returns as percentage changes + 1
cumreturns = groreturns.cumprod() # calculate the cumulative gross returns
wealth = cumreturns.dot(weight).to_frame() # calculate the portfolio value = cummulative gross returns
porreturns = wealth.pct_change().dropna() # calculate the portfolio returns
# we do not consider transaction costs - acctualy, as we buy and hold, there would occur transaction costs only two times: at the beginning when we buy the portoflio and at the end when we sell the portfolio


# visualize 
fig, (ax1, ax2) = plt.subplots(2, sharex=True) # create the figure with 2 plots/graphs
fig.suptitle('Naive portoflio evolution') # add the title to the figure
ax1.plot(wealth[0]) # in the first graph plot the value of the portfolio in time
ax2.plot(porreturns[0]) # in the second graph plot the (daily) returns 
ax1.set_ylabel('Value') # set the label of y axis
ax2.set_ylabel('Return') # set the label of y axis
del ax1, ax2  # delete unnecessary variables

#--------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


wealth.rename(columns={wealth.columns[0]: "naive" }, inplace = True) # rename the first to column to "naive"
"""