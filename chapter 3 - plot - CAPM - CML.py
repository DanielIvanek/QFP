import numpy as np
import matplotlib.pyplot as plt

# Define parameters
eR = np.array([0.08, 0.04])  # Expected returns for the two stocks
sigma = np.array([0.4, 0.1])  # Standard deviations for the two stocks
R = 0.1  # Correlation coefficient between the two stocks
RF = 0.05  # Risk-free rate

# Generate portfolios on the efficient frontier
portfolios = []
for w1 in np.arange(0, 1.01, 0.01):  # Loop over weight of the first stock
    w2 = 1 - w1  # Calculate weight of the second stock
    ret = eR[0] * w1 + eR[1] * w2  # Portfolio expected return
    sig = ((w1 * sigma[0])**2) + w1 * w2 * sigma[0] * sigma[1] * R + (w2 * sigma[1])**2  # Portfolio standard deviation
    portfolios.append([ret, sig])

portfolios = np.array(portfolios)  # Convert the portfolios list to a NumPy array

# Remove the points where the standard deviation is decreasing
decreasing_std_indices = np.where(np.diff(portfolios[:, 1]) < 0)[0]
portfolios = np.delete(portfolios, decreasing_std_indices, axis=0)

# Find the portfolio with maximum slope (Tangent Portfolio)
cml_slope = (portfolios[:, 0] - RF) / portfolios[:, 1]
tangent_portfolio_index = np.argmax(cml_slope)
tangent_portfolio = portfolios[tangent_portfolio_index]
sharpe = cml_slope[tangent_portfolio_index]

# Plotting
plt.figure(figsize=(10, 6))

# Plot efficient frontier
plt.plot(portfolios[:, 1], portfolios[:, 0], label='Markowitz Efficient Frontier', color='blue', linewidth=2)

# Plot Capital Market Line (CML) - Lending Portfolios
x = np.linspace(0, portfolios[tangent_portfolio_index, 1], 100)
y = x * sharpe + RF
plt.plot(x, y, label='Capital Market Line (CML) - Lending Portfolios', color='green', linestyle='-', linewidth=3)

# Plot Capital Market Line (CML) - Borrowing Portfolios
x = np.linspace(portfolios[tangent_portfolio_index, 1], portfolios[-1, 1], 100)
y = x * sharpe + RF
plt.plot(x, y, label='Capital Market Line (CML) - Borrowing Portfolios', color='red', linestyle='-', linewidth=3)

# Plot the Tangent Portfolio (Market Portfolio)
plt.scatter(tangent_portfolio[1], tangent_portfolio[0], color='black', marker='o', label='Tangent Portfolio (Market Portfolio)')
plt.scatter(0, RF, color='green', marker='o', label='Risk-free portfolio')


#plt.text(0.005, 0.052, 'Risk-free rate', verticalalignment='bottom', horizontalalignment='right', color='black', fontsize=10, rotation='vertical')
#plt.text(tangent_portfolio[1], tangent_portfolio[0] - 0.002, 'Tangential Portfolio', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=10)

arrow_props = dict(facecolor='black', edgecolor='black', shrink=0.05, width=0.1)  # Adjust linewidth for arrows
plt.annotate('Risk-free portfolio', [0, RF],[0.01, 0.06], arrowprops=arrow_props, rotation='vertical', va='bottom')
plt.annotate("Tangential Portfolio", [tangent_portfolio[1], tangent_portfolio[0]],[0.07, 0.06], arrowprops=arrow_props)


# Adding labels and legend
plt.title('Efficient Frontier and Capital Market Line (CML)')
plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Expected Return')
plt.legend()

# Set y-axis and x-axis lower limits to 0
# plt.ylim(bottom=0)
plt.xlim(left=0)
# Display the grid
#plt.grid(True)

# Show the plot
plt.show()
