import numpy as np

def BOPM_option(S, K, T, r, sigma, option_type, num_steps=100):
    # First, we calculate all the parameters of the binomial model:
    delta_t = T / num_steps
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)

    # Generate binomial tree for underlying asset price
    underlying_tree = np.zeros((num_steps + 1, num_steps + 1))
    for i in range(num_steps + 1):
        for j in range(i + 1):
            underlying_tree[j, i] = S * u**(i-j) * d**j

    # We evaluate the option for individual nodes of the binomial tree at maturity time
    option_values = np.zeros((num_steps + 1, num_steps + 1))
    if "call" in option_type:
        option_values[:, -1] = np.maximum(0, underlying_tree[:, -1] - K)
    elif "put" in option_type:
        option_values[:, -1] = np.maximum(0, K - underlying_tree[:, -1])
    else:
        raise ValueError("Invalid option type. Use 'European call', 'European put', 'American call', or 'American put'.")

    # Option valuation using binomial tree (backward induction)
    for i in range(num_steps - 1, -1, -1):
        for j in range(i+1):
            option_values[j, i] = np.exp(-r * delta_t) * (p * option_values[j, i + 1] + (1 - p) * option_values[j+1, i + 1])
            if option_type.startswith("American"):
                # Check for early exercise
                early_exercise_value = np.maximum(0, K - underlying_tree[j, i]) if "put" in option_type else np.maximum(0, underlying_tree[j, i] - K)
                option_values[num_steps - j, i] = max(option_values[num_steps - j, i], early_exercise_value)
    return option_values[0, 0]

print("European Call Option Price:", BOPM_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="European call"))
print("European Put Option Price:", BOPM_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="European put"))
print("American Call Option Price:", BOPM_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="American call"))
print("American Put Option Price:", BOPM_option(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="American put"))
