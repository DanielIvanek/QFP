import yfinance as yf  # we import the library and rename it to yf
from scipy.stats import chi2, norm, multivariate_normal, multivariate_t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


##################################################################################
# VaR and CVaR estimation methods - definiton of functions 
##################################################################################

# calculates VaR and CVaR by means of analytical method assuming normal distribution
def calcAnalyticalSingle(ret, alpha): # single risk factors
    VaR = norm.ppf(1-alpha) * ret.std() - ret.mean() # we calculate VaR
    CVaR = norm.pdf(norm.ppf(alpha))/(alpha) * ret.std() - ret.mean() # we calculate CVaR
    return VaR.iloc[0], CVaR.iloc[0]


def calcAnalyticalMultiple(ret, w, alpha): # multiple risk factors
    mu = ret.mean() # we calculate expected returns
    Q = ret.cov()     # we calculate covariance matrix
    stdp = np.sqrt(w.transpose().dot(Q.dot(w))) # we calculate portfolio standard deviation
    mup = mu.transpose().dot(w) # we calculate portfolio return expected value
    VaR = norm.ppf(1-alpha) * stdp - mup # we calculate VaR
    CVaR = norm.pdf(norm.ppf(alpha))/(alpha) * stdp - mup # we calculate CVaR
    return VaR.iloc[0,0], CVaR.iloc[0,0]


# calculates VaR and CVaR by means of historical simulation method
def calcHistoricalSingle(ret, alpha): # single risk factors
     VaR = -ret.quantile(q=alpha, interpolation='linear') 
     CVaR = -ret[ret<-VaR].mean()
     return VaR.iloc[0], CVaR.iloc[0]


def calcHistoricalMultiple(ret, w, alpha): # multiple risk factors
    # calculates VaR by means of historical simulation
    returns_historical = ret.dot(w) # we calculate new/future portfolio returns if the historical returns repeat 
    VaR = -returns_historical.quantile(q=alpha, interpolation='linear') 
    CVaR = -returns_historical[returns_historical<-VaR].mean()
    return VaR.iloc[0], CVaR.iloc[0]


# calculates VaR and CVaR by means of Monte Carlo simulation method assuming normal distribution 
def calcMCSMultiple_n(ret, w, alpha):
    # calculates VaR by means of Monte Carlo simulation assuming joint normal dist.
    mu = ret.mean() # we calculate mean/expected returns
    Q = ret.cov() # we calculate covariance matrix
    returns_simulated = multivariate_normal.rvs(mean=mu, cov=Q, size=1000000, random_state=100)
    returns_simulated = pd.DataFrame(returns_simulated, columns=ret.columns).dot(w)
    VaR = -returns_simulated.quantile(q=alpha, interpolation='linear') 
    CVaR = -returns_simulated[returns_simulated<-VaR].mean()
    return VaR.iloc[0], CVaR.iloc[0]


# calculates VaR and CVaR by means of Monte Carlo simulation method assuming student t distribution 
def calcMCSMultiple_t(ret, w, alpha, df):
    # calculates VaR by means of Monte Carlo simulation assuming joint Student distribution with df degrees of freedom
    mu = ret.mean() # we calculate mean/expected returns
    Q = ret.cov() # we calculate covariance matrix
    returns_simulated = multivariate_t.rvs(loc=mu, shape=Q, df=df, size=100000, random_state=100)
    returns_simulated = pd.DataFrame(returns_simulated, columns=ret.columns).dot(w)
    VaR = -returns_simulated.quantile(q=alpha, interpolation='linear') 
    CVaR = -returns_simulated[returns_simulated<-VaR].mean()
    return VaR.iloc[0], CVaR.iloc[0]

##################################################################################
# Main part calculating the VaR a CVaR using the functions above
##################################################################################
tickers = ["META",  # Facebook/Meta Platforms, Inc.
           "AAPL",  # Apple
           "AMZN",  # Amazon
           "NFLX",  # Netflix
           "GOOGL"  # Google (Alphabet Inc.)
          ]
alpha=0.05

# we download the data
prices = yf.download(tickers, start="2018-01-01", end="2022-12-31", interval = "1d", group_by="column", auto_adjust = True,)["Close"]
returns = prices.pct_change().dropna() # we can calculate percentage changes (simple/discrete returns)

v = pd.DataFrame(data=[1, 1, 1, 1, 1], index=tickers, columns=['weight']) # quantitites held
u = v * prices.iloc[-1:].transpose().values  # weights in absolute value (in $)
W = u.sum()  # portoflio value
w = u /W # relative weights (in %)

print(f'Portfolio value(W): {W[0]:.2f}')
print(f'Portfolio absolute weights(u): {u}')
print(f'Portfolio relative weights(u): {w}')

"""
tickers = ["PG", "TSLA"]  # alphabetically sort :-)
alpha=0.05

# we download the data
prices = yf.download(tickers, start="2010-01-01", end="2022-12-31",interval = "1d", group_by="column", auto_adjust = True,)["Close"]
returns = prices.pct_change().dropna() # we can calculate percentage changes (simple/discrete returns)

v = pd.DataFrame(data=[1, 1], index=tickers, columns=['weight']) # quantitites held
u = v * prices.iloc[-1:].transpose().values  # weights in absolute value (in $)
w = u /u.sum() # relative weights (in %)
W = u.sum()  # portoflio value
"""

""" Analytical method """
# Calculate and print the relative VaR and CVaR 
relative_VaR, relative_CVaR = calcAnalyticalSingle(returns.dot(w), alpha)
print(f"Relative VaR: {relative_VaR:.6f}, relative CVaR: {relative_CVaR:.6f}")
relative_VaR, relative_CVaR = calcAnalyticalMultiple(returns, w, alpha)
print(f"Relative VaR (Multiple): {relative_VaR:.6f}, relative CVaR (Multiple): {relative_CVaR:.6f}")

# Calculate and print the absolute VaR and CVaR 
relative_VaR, relative_CVaR = calcAnalyticalSingle(returns.dot(w), alpha)
absolute_VaR = W.values[0] * relative_VaR
absolute_CVaR = W.values[0] * relative_CVaR
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
absolute_VaR, absolute_CVaR = calcAnalyticalSingle(returns.dot(u), alpha)
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
relative_VaR, relative_CVaR = calcAnalyticalMultiple(returns, w, alpha)
absolute_VaR = W.values[0] * relative_VaR
absolute_CVaR = W.values[0] * relative_CVaR
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
absolute_VaR, absolute_CVaR = calcAnalyticalMultiple(returns, u, alpha)
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")


""" Historical simulation method """
# Calculate and print the relative VaR and CVaR 
relative_VaR, relative_CVaR = calcHistoricalSingle(returns.dot(w), alpha)
print(f"Relative VaR: {relative_VaR:.6f}, relative CVaR: {relative_CVaR:.6f}")
relative_VaR, relative_CVaR = calcHistoricalMultiple(returns, w, alpha)
print(f"Relative VaR (Multiple): {relative_VaR:.6f}, relative CVaR (Multiple): {relative_CVaR:.6f}")

# Calculate and print the absolute VaR and CVaR 
relative_VaR, relative_CVaR = calcHistoricalSingle(returns.dot(w), alpha)
absolute_VaR = W.values[0] * relative_VaR
absolute_CVaR = W.values[0] * relative_CVaR
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
absolute_VaR, absolute_CVaR = calcHistoricalSingle(returns.dot(u), alpha)
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
relative_VaR, relative_CVaR = calcHistoricalMultiple(returns, w, alpha)
absolute_VaR = W.values[0] * relative_VaR
absolute_CVaR = W.values[0] * relative_CVaR
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
absolute_VaR, absolute_CVaR = calcHistoricalMultiple(returns, u, alpha)
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")


""" Monte Carlo - normal distribution"""
# Calculate and print the relative VaR and CVaR 
relative_VaR, relative_CVaR = calcMCSMultiple_n(returns, w, alpha)
print(f"Relative VaR: {relative_VaR:.6f}, relative CVaR: {relative_CVaR:.6f}")

# Calculate and print the absolute VaR and CVaR 
relative_VaR, relative_CVaR = calcMCSMultiple_n(returns, w, alpha)
absolute_VaR = W.values[0] * relative_VaR
absolute_CVaR = W.values[0] * relative_CVaR
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")
absolute_VaR, absolute_CVaR = calcMCSMultiple_n(returns, u, alpha)
print(f"Absolute VaR: {absolute_VaR:.6f}, absolute CVaR: {absolute_CVaR:.6f}")


""" Monte Carlo - joint Student distribution"""
# Calculate and print the relative VaR and CVaR 
relative_VaR, relative_CVaR = calcMCSMultiple_t(returns, w, alpha, df=4)
print(f"Degrees of freedom 4. Relative VaR: {relative_VaR:.6f}, relative CVaR: {relative_CVaR:.6f}")
relative_VaR, relative_CVaR = calcMCSMultiple_t(returns, w, alpha, df=40)
print(f"Degrees of freedom 40. Relative VaR: {relative_VaR:.6f}, relative CVaR: {relative_CVaR:.6f}")
relative_VaR, relative_CVaR = calcMCSMultiple_t(returns, w, alpha, df=400)
print(f"Degrees of freedom 400. Relative VaR: {relative_VaR:.6f}, relative CVaR: {relative_CVaR:.6f}")




##################################################################################
# Backtesting
##################################################################################
def kupiecTest(I, p):
    n = len(I)  # the length of backtested series 
    n1 = sum(I) # no. violations
    n0 = n-n1   # no. nonviolations
    kupiec = -2 * n0 * np.log(1-p) - 2 * n1 * np.log(p)  # the first part of the formula
    if n0 > 0: # add the second part of the formula only if n0>0, otherwise log(0) would cause an error
        kupiec = kupiec + 2 * n0 * np.log(n0/n)
    if n1 > 0: # add the third part of the formula only if n1>0, otherwise log(0) would cause an error
        kupiec = kupiec + 2* n1 * np.log(n1/n) 
    pvalue = 1-chi2.cdf(kupiec, 1) # calculate p-value, note that the null hypothesis is that the observed and expected probability of violation occurring, i.e. we want the p-value to be as high as possible
    return pvalue


alphas = [0.001, 0.005, 0.01, 0.05, 0.10, 0.15]

m = 250 # the length of in-sample period
n = len(returns) - m # the lengt of backtesting period

# inizicalize the dataframes for storing the results
exceptions = pd.DataFrame(columns = alphas, index=['analytical', 'historical', 'Monte Carlo', 'expected']) # prepare empty dataframe for results
pValueKupiec = pd.DataFrame(columns = alphas, index=['analytical', 'historical', 'Monte Carlo']) # prepare empty dataframe for results
    

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12.5, 17))  # Create a grid of 3x2 subplots
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

for i, alpha in enumerate(alphas):
    VaRs = pd.DataFrame(index = returns.index[m:], columns=['analytical', 'historical', 'Monte Carlo', 'True loss']) # create an empty DataFrame
    for t in range(m, m+n):
        inSample = returns[t-m:t-1] # define in-sample period for VaR estimation
        outOfSampleReturns = returns[t:t+1] # define out-of-sample period for VaR estimate verification
   
        VaRs.iloc[t-m,0], _ = calcAnalyticalMultiple(inSample, w, alpha) # calculate VaR estimate via analytical method 
        VaRs.iloc[t-m,1], _ = calcHistoricalMultiple(inSample, w, alpha) # calculate VaR estimate via historical simulation method 
        VaRs.iloc[t-m,2], _ = calcMCSMultiple_t(inSample, w, alpha, 4)  # calculate VaR estimate via Monte Carlo simulation method assuming student t-distribution with 3 df
        VaRs.iloc[t-m,3] = -pd.DataFrame(data=outOfSampleReturns).dot(w).iloc[0,0] # calculate true observed loss
    
        print(f"Cycle {t-m+1} out of {n} for alpha={alpha} done.")
    
    for column in VaRs.columns[0:-1]:
        I = VaRs['True loss'] > VaRs[column] # get the series of exceptions (VaR violations)
        exceptions[alpha][column] = sum(I) # get the number of exceptions (VaR violations)
        pValueKupiec[alpha][column] = kupiecTest(I, alpha) # calculate the p-value of Kupiec test
    exceptions[alpha]['expected'] = alpha * n # calculate the expected number of exceptions

    ax = axes[i]
    VaRs['x'] = VaRs.index  # Add new column generated from the index to plot the figures
    VaRs.plot.scatter(x='x', y='True loss', marker='.', ax=ax)  # Plot the true losses on the ith subplot
    VaRs.plot(y=['analytical', 'historical', 'Monte Carlo'], ax=ax)  # Add the VaR estimates on the ith subplot
    #ax.set_xticklabels(VaRs['x'], rotation=90)  # Rotate the labels of x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks to be at the start of each year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format major ticks to show only the year
    ax.set_ylim(bottom=0)  # Set the lower limit of y-axis to 0
    ax.set_title(f'VaRs for alpha={alpha}')  # Set the title of the ith subplot
    ax.set_ylabel('VaRs & True Loss')  # Set the label of y-axis
    ax.set_xlabel('Date')  # Set the label of x-axis
    ax.legend(loc="best")    


plt.tight_layout()  # Adjust subplots to fit into the figure area
plt.show()  # Display the figure with 6 subplots
    
print(exceptions)
print(pValueKupiec)
    

"""
print(exceptions)
             0.001  0.005 0.010  0.050 0.100  0.150
analytical       8      9    10     25    36     52
historical       6      7     9     26    49     77
Monte Carlo      0      2     4     10    25     37
Expected     0.505  2.525  5.05  25.25  50.5  75.75

print(pValueKupiec)
                0.001     0.005     0.010     0.050     0.100     0.150
analytical        0.0  0.001556  0.050853  0.959226  0.024078  0.001877
historical   0.000015  0.020541  0.111534  0.878853  0.823155  0.876483
Monte Carlo  0.314782  0.731025  0.626052  0.000417  0.000033       0.0    
    
# As can be seen the variance-covariance approach works very well for 5% significance level (original proposal by JP Morgan)
# For low alphas, normal distribution underestimates the risk. For alhpas higher than 5%, normal distribution can underestimate the risk.
# Historical simulation works relatively well
"""



