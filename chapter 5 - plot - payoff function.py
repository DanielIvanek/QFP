import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the payoff of a put option
def put_payoff(S_T, strike):
    return np.maximum(strike - S_T, 0)

# Function to calculate the payoff of a call option
def call_payoff(S_T, strike):
    return np.maximum(S_T - strike, 0)

# Function to calculate the total profit (payoff - option premium)
def total_profit(payoff, option_premium):
    return payoff - option_premium

# Generate a range of underlying prices at maturity (S_T)
S_T_range = np.linspace(80, 120, 100)

# Set the strike price and option premium
strike_price = 100
option_premium = 5

# Calculate the payoffs for put and call options
put_payoffs = put_payoff(S_T_range, strike_price)
call_payoffs = call_payoff(S_T_range, strike_price)

# Calculate the total profits
put_profits = total_profit(put_payoffs, option_premium)
call_profits = total_profit(call_payoffs, option_premium)

# Create a figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Top-left subplot: Put option payoff and profit (Long position)
axs[0, 0].plot(S_T_range, put_payoffs, label='Put Option Payoff (Long)')
axs[0, 0].plot(S_T_range, put_profits, label='Put Option Profit (Long)', linestyle='--')
axs[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axs[0, 0].axvline(strike_price, color='red', linestyle='--', linewidth=1, label='Put Option Strike')
axs[0, 0].set_title('Put Option Payoff and Profit (Long)')
axs[0, 0].set_xlabel('Underlying Price at Maturity (S_T)')
axs[0, 0].set_ylabel('Payoff / Profit')
axs[0, 0].legend()

# Top-right subplot: Call option payoff and profit (Long position)
axs[0, 1].plot(S_T_range, call_payoffs, label='Call Option Payoff (Long)')
axs[0, 1].plot(S_T_range, call_profits, label='Call Option Profit (Long)', linestyle='--')
axs[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axs[0, 1].axvline(strike_price, color='red', linestyle='--', linewidth=1, label='Call Option Strike')
axs[0, 1].set_title('Call Option Payoff and Profit (Long)')
axs[0, 1].set_xlabel('Underlying Price at Maturity (S_T)')
axs[0, 1].set_ylabel('Payoff / Profit')
axs[0, 1].legend()

# Bottom-left subplot: Put option payoff and profit (Short position)
axs[1, 0].plot(S_T_range, -put_payoffs, label='Put Option Payoff (Short)')
axs[1, 0].plot(S_T_range, -put_profits, label='Put Option Profit (Short)', linestyle='--')
axs[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axs[1, 0].axvline(strike_price, color='red', linestyle='--', linewidth=1, label='Put Option Strike')
axs[1, 0].set_title('Put Option Payoff and Profit (Short)')
axs[1, 0].set_xlabel('Underlying Price at Maturity (S_T)')
axs[1, 0].set_ylabel('Payoff / Profit')
axs[1, 0].legend()

# Bottom-right subplot: Call option payoff and profit (Short position)
axs[1, 1].plot(S_T_range, -call_payoffs, label='Call Option Payoff (Short)')
axs[1, 1].plot(S_T_range, -call_profits, label='Call Option Profit (Short)', linestyle='--')
axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axs[1, 1].axvline(strike_price, color='red', linestyle='--', linewidth=1, label='Call Option Strike')
axs[1, 1].set_title('Call Option Payoff and Profit (Short)')
axs[1, 1].set_xlabel('Underlying Price at Maturity (S_T)')
axs[1, 1].set_ylabel('Payoff / Profit')
axs[1, 1].legend()

# Display the plot
plt.tight_layout()
plt.show()


