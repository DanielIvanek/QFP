import numpy as np

def MC_option(S, K, T, r, sigma, option_type, num_trials=10000, num_steps=100):
    np.random.seed(100) # fix the seed
    
    mu = r # under risk-neutral probability mu=r
    dt = T / num_steps  
    
    # Simulate stock price paths using geometric Brownian motion
 
    # Initialize an array to store stock prices for each trial
    stock_prices = np.zeros((num_trials, num_steps + 1))
    stock_prices[:, 0] = S
 
    # Simulate stock prices for multiple trials using geometric Brownian motion
    for i in range(1, num_steps + 1):
        drift = (mu - (sigma**2)/2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1, num_trials)
        stock_prices[:, i] = stock_prices[:, i - 1] * np.exp(drift + diffusion)
 
    # Calculate option payoffs at maturity
    if option_type == 'call':
        option_payoffs = np.maximum(stock_prices[:, -1] - K, 0)
    elif option_type == 'put':
        option_payoffs = np.maximum(K - stock_prices[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Discount the expected payoffs to present value
    option_price = np.mean(option_payoffs * np.exp(-r * T))

    return option_price

print("European Call Option Price:", MC_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"))
print("European Put Option Price:", MC_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"))
