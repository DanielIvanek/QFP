import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
mu = 0.1  # Drift
sigma = 0.2  # Volatility
S0 = 100  # Initial stock price
dt = 0.01  # Time step
T = 1  # Total time
num_trials = 10000  # Number of simulation trials

# Number of time steps
num_steps = int(T / dt)

# fix the seed
np.random.seed(100)

# Initialize a DataFrame to store stock prices for each trial
stock_prices_df = pd.DataFrame(index=range(num_trials), columns=range(num_steps + 1))
stock_prices_df.iloc[:, 0] = S0

# Simulate stock prices for multiple trials using geometric Brownian motion
for i in range(1, num_steps + 1):
    drift = (mu - (sigma**2)/2) * dt
    diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1, num_trials)
    stock_prices_df.iloc[:, i] = stock_prices_df.iloc[:, i - 1] * np.exp(drift + diffusion)

plt.figure(figsize=(10, 6))

# Plot the simulated stock prices for the first 10 trials
ax1 = plt.subplot(1,2,1)
for i in range(10):
    plt.plot(stock_prices_df.columns * dt, stock_prices_df.iloc[i, :], label=f'Trial {i + 1}')
plt.title('Stock Price Simulation with Pandas (First 10 Trials)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

# plot the histogram of the last prices (at time T) with horizontal orientation
plt.subplot(1,2,2, sharey=ax1)
plt.hist(stock_prices_df.iloc[:, -1], bins=100, color='blue', alpha=0.7, edgecolor='black', orientation='horizontal')
plt.title('Histogram of Last Prices from Stock Price Simulation')
plt.ylabel('Stock Price')
plt.xlabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Print mean and standard deviation of the prices at the end
print(f"Mean of the prices at the end: {stock_prices_df.iloc[:, -1].mean()}")
print(f"Standard deviation of the prices at the end: {stock_prices_df.iloc[:, -1].std()}")