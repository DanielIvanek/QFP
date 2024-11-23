import numpy as np

def LSMC_option(S, K, T, r, sigma, option_type, num_trials=10000, num_steps=100):
    # Function to calculate the payoff of the option
    def payoff(S, K, option_type):
        if "call" in option_type:
            return np.maximum(0, S - K)
        elif "put" in option_type:
            return np.maximum(0, K - S)
        else:
            raise ValueError("Invalid option type. Use 'European call', 'European put', 'American call', or 'American put'.") 
    
    np.random.seed(100) # Fix the seed for reproducibility
    
    mu = r  # under risk-neutral probability mu=r
    dt = T / num_steps  

    # Simulate stock price paths using geometric Brownian motion
    stock_prices = np.zeros((num_trials, num_steps + 1))
    stock_prices[:, 0] = S

    for t in range(1, num_steps + 1):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1, num_trials)
        stock_prices[:, t] = stock_prices[:, t - 1] * np.exp(drift + diffusion)
    
    # Matrix to store whether the option is exercised at each time step
    excercise = np.zeros((num_trials, num_steps + 1)) 
    
    # Matrix to store the cash flows of the option at each time step
    CFs = np.zeros((num_trials, num_steps + 1)) 
    CFs[:, -1] = payoff(stock_prices[:, -1], K, option_type)

    if option_type.startswith("European"):
        # European Option Pricing
        return np.mean(CFs[:, -1] * np.exp(-r * T))
    
    elif option_type.startswith("American"):
        # American Option Pricing using Least Squares Monte Carlo
        for t in range(num_steps - 1, -1, -1):
            # Calculate payoffs
            payoffs = payoff(stock_prices[:, t], K, option_type)
          
            # Find paths where the option is in-the-money
            hold = np.where(payoffs > 0)[0]  
            
            if len(hold) > 3:
                # Apply Least square method of future CF on current stock price
                regression = np.polyfit(stock_prices[hold, t].flatten(), CFs[hold, t+1].flatten()*np.exp(-r * dt), 3)
                # calculate continuation values for ITM options
                CV = np.polyval(regression, stock_prices[hold, t].flatten())

                # Check whether to exercise the option now or continue holding it
                excercise[hold, t] = np.where(payoffs[hold] >= CV, 1, 0)
                # Update cash flows based on exercise decision
                CFs[:, t] = np.where(excercise[:, t].flatten()==1, payoffs, CFs[:, t+1].flatten()*np.exp(-r * dt))
            else:
                # If there are very few valid paths with ITM options, do not perform regression
                CFs[:, t] = CFs[:, t+1].flatten()*np.exp(-r * dt)

        return np.mean(CFs[:, 0])
             
    else:
        raise ValueError("Invalid option type. Use 'European call', 'European put', 'American call', or 'American put'.") 


# Example usage
print("European Call Option Price:", LSMC_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="European call"))
print("European Put Option Price:", LSMC_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="European put"))
print("American Call Option Price:", LSMC_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="American call"))
print("American Put Option Price:", LSMC_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="American put"))
