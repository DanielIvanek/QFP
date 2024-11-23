
# --------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm

def BS_option(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + (sigma**2)/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type == "put":
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")  

print("Call Option Price:", BS_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"))
print("Put Option Price:", BS_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"))

# --------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm

def BS_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
  
    if option_type == "call":
        delta = norm.cdf(d1)  # Delta
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Gamma
        vega = S * np.sqrt(T) * norm.pdf(d1)  # Vega
        theta = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)  # Theta
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)  # Rho
    elif option_type == "put":
        delta = norm.cdf(d1) - 1  # Delta for a put option
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Gamma for a put option
        vega = S * np.sqrt(T) * norm.pdf(d1)  # Vega for a put option
        theta = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)  # Theta for a put option
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)  # Rho for a put option
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return delta, gamma, vega, theta, rho

delta_call, gamma_call, vega_call, theta_call, rho_call = BS_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
delta_put, gamma_put, vega_put, theta_put, rho_put = BS_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")

print("Call Option Greeks:")
print("Delta:", delta_call)
print("Gamma:", gamma_call)
print("Vega:", vega_call)
print("Theta:", theta_call)
print("Rho:", rho_call)

print("\nPut Option Greeks:")
print("Delta:", delta_put)
print("Gamma:", gamma_put)
print("Vega:", vega_put)
print("Theta:", theta_put)
print("Rho:", rho_put)

# --------------------------------------------------------------------------------------------------------------------------------------------

# ------
def BS_iv(price, S, K, T, r, option_type='call'):
    def fun(sigma):
        #return abs(BS_option(S, K, T, r, sigma, option_type) - price)
        return (BS_option(S, K, T, r, sigma, option_type) / price - 1) ** 2

    res = scipy.optimize.minimize_scalar(fun, bounds=(0.001, 6), method='bounded')
    return res.x

iv_call = BS_iv(price=10.45058357, S=100, K=100, T=1, r=0.05, option_type="call")
iv_put = BS_iv(price=5.573526022, S=100, K=100, T=1, r=0.05, option_type="put")
print("Implied volatility of call option: ", iv_call)
print("Implied volatility of put option: ", iv_put)
#--------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scipy
import datetime as dt

import yfinance as yf
import matplotlib.pyplot as plt

# https://finance.yahoo.com/news/read-options-table-080007410.html

r = 0.05

stock = yf.Ticker("META") # define the ticker for which we want to download the data; we chose META as it does not pay dividends
expirations = stock.options # get the list of possible expiration dates data
print(expirations[-1])

calls = stock.option_chain(expirations[-1]).calls # get all the call options for the expiration data third from the last one
calls.insert(loc=6, column='price', value=(calls['ask']+calls['bid'])/2) # if traded and the ask,bid is known
#calls.insert(loc=6, column='price', value=(calls['lastPrice'])) # if ask bid prices are not available 
S = stock.history(period='1d', interval='1m')['Close'].iloc[-1] # we get the last price of the underlaying asset(stock)   

ttm = (pd.to_datetime(expirations[-1]) - dt.datetime.now()).total_seconds() /60/60/24/252
calls.insert(8, 'our IV', np.nan)

for index, row in calls.iterrows():
    calls['our IV'].values[index]=BS_iv(row['price'], S, row['strike'], ttm, r, 'call')

plt.scatter(x=calls['strike'], y=calls['our IV'])
plt.xlabel('Strike Price')
plt.ylabel('Our IV')
plt.title('Our IV vs. Strike Price for Call Options')
plt.show()

    
    

# further thoughts:
# volatility surface
# Least squares Monte Carlo (LSM) method: https://medium.datadriveninvestor.com/a-complete-step-by-step-guide-for-pricing-american-option-712c84aa254e