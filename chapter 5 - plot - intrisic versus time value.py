import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes Call Option Pricing Function
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Black-Scholes Put Option Pricing Function
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Payoff functions
def call_payoff(S, K):
    return np.maximum(S - K, 0)

def put_payoff(S, K):
    return np.maximum(K - S, 0)

# Parameters
K = 100  # Strike Price
sigma = 0.2  # Volatility
r = 0.08  # Risk-free Rate

# Time to maturity (τ) and underlying prices (S0) ranges
time_to_maturity_range = [0.25, 0.5, 0.75, 1.0]
S0_range = np.linspace(80, 120, 100)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Left subplot: Call option premiums and payoff
for T in time_to_maturity_range:
    call_prices = [black_scholes_call(S0, K, T, r, sigma) for S0 in S0_range]
    axs[0].plot(S0_range, call_prices, label=f'Premium (Time to Maturity={T:.2f})')
    
call_payoffs = [call_payoff(S0, K) for S0 in S0_range]
axs[0].plot(S0_range, call_payoffs, linestyle='--', label='Intrinsic value (Payoff)')
axs[0].set_title('European Call Option Premiums')
axs[0].set_xlabel('Underlying Price (S0)')
axs[0].set_ylabel('Option Premium')
axs[0].legend()

# Right subplot: Put option premiums and payoff
for T in time_to_maturity_range:
    put_prices = [black_scholes_put(S0, K, T, r, sigma) for S0 in S0_range]
    axs[1].plot(S0_range, put_prices, label=f'Premium (Time to Maturity={T:.2f})')

put_payoffs = [put_payoff(S0, K) for S0 in S0_range]
axs[1].plot(S0_range, put_payoffs, linestyle='--', label='Intrinsic value (Payoff)')
axs[1].set_title('European Put Option Premiums')
axs[1].set_xlabel('Underlying Price (S0)')
axs[1].set_ylabel('Option Premium')
axs[1].legend()

plt.tight_layout()
plt.show()
