import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model


# Download INTC stock data and calculate log returns
data = yf.download('INTC', start='2010-01-01', end='2022-01-01', auto_adjust=True)
close_prices = data['Close']
log_returns = 100 * np.log(close_prices / close_prices.shift(1)).dropna()

# Estimate AR(1)-GARCH(1,1) model
model = arch_model(log_returns, mean='AR', lags=1, vol='Garch', p=1, q=1)
results = model.fit()
print(results.summary()) # Print the summary of the model


# Simulate 1000 daily returns according to fitted parameters
sim_mod = arch_model(None, mean='AR', lags=1, vol='Garch', p=1, q=1)
sim_data = sim_mod.simulate(results.params, 1000)
sim_data.head()

# Extend the index of datafame by adding another 1000 business days
sim_data.index = pd.bdate_range(log_returns.index[-1], periods=1001)[1:]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
# Plot returns
ax1.plot(log_returns.index, log_returns, color='blue', label='Observed Returns')
ax1.plot(sim_data.index, sim_data['data'], color='red', label='Simulated Returns')
ax1.set_title('Returns (Observed and Simulated)')
ax1.legend()

# Plot volatilities
ax2.plot(log_returns.index, results.conditional_volatility, color='blue', label='Observed Volatility')
ax2.plot(sim_data.index, sim_data['volatility'], color='red', label='Simulated Volatility')
ax2.set_title('Volatility (Observed and Simulated)')
ax2.legend()

plt.tight_layout()
plt.show()


# Different model cpecifications with N(0,1) distribution 
model = arch_model(log_returns) # GARCH(1,1) without AR() process
model = arch_model(log_returns, p=1, o=1, q=1) #GJR-GARCH model without AR() process
model = arch_model(log_returns, p=1, o=1, q=1, power=1.0) # TARCH(1,1) (also known as ZARCH) model without AR() process

# Different model cpecifications with Student t distribution 
model = arch_model(log_returns, dist="StudentsT") # GARCH(1,1) without AR() process
model = arch_model(log_returns, p=1, o=1, q=1, dist="StudentsT") #GJR-GARCH model without AR() process
model = arch_model(log_returns, p=1, o=1, q=1, power=1.0, dist="StudentsT") # TARCH(1,1) (also known as ZARCH) model without AR() process