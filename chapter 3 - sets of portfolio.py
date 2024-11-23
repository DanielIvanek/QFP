import numpy as np
import matplotlib.pyplot as plt

np.random.seed(150)

# Number of assets
n_assets = 3


# Randomly generated covariance matrix for n assets
# For simplicity, we'll use a symmetric and positive definite matrix
expected_returns = np.array([0.0186, 0.0498, 0.01])
cov_matrix =  np.array([[ 0.008464, -0.002644, 0.000387],
                        [-0.002644,  0.059195, 0.004045],
                        [ 0.000387,  0.004045, 0.017822]
                       ])


# Number of portfolios to simulate
n_portfolios = 200000

# Initialize arrays to store weights, returns, and risks
portfolio_returns = []
portfolio_risks = []

for _ in range(n_portfolios):
    # Generate random weights for n assets and normalize them to sum to 1
    weights = np.random.rand(n_assets)
    weights /= np.sum(weights)
    
    # Expected portfolio return
    portfolio_returns.append(np.dot(weights, expected_returns))
    
    # Portfolio standard deviation (risk)
    portfolio_risks.append(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

# Convert lists to numpy arrays
portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)

# Sort the portfolios by risk
sorted_indices = np.argsort(portfolio_risks)
sorted_risks = portfolio_risks[sorted_indices]
sorted_returns = portfolio_returns[sorted_indices]

# Identify the efficient frontier
efficient_risks = []
efficient_returns = []
current_max_return = 0
for risk, ret in zip(sorted_risks, sorted_returns):
    if ret >= current_max_return:
        efficient_risks.append(risk)
        efficient_returns.append(ret)
        current_max_return = ret


# Plot the feasible set of portfolios
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_risks, portfolio_returns, c='lightgray', marker='o', label='feasible portoflios')
# Highlight the efficient frontier
plt.plot(efficient_risks, efficient_returns, color='darkgray', linewidth=5, label='efficient portfolios')
plt.scatter(0.1, 0.02, c='red', label='Portfolio A', zorder=5) 
plt.scatter(0.0722, 0.02, c='blue', label='Portfolio B', zorder=5)
plt.scatter(0.1, 0.03, c='green', label='Portfolio C', zorder=5)
plt.scatter(0.0806, 0.025, c='orange', label='Portfolio D', zorder=5)
plt.title('Feasible Set of Portfolios and Efficient Frontier')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Expected Return')
plt.legend()
plt.tight_layout()
plt.show()