import numpy as np
import matplotlib.pyplot as plt

# Define parameters
betas = np.arange(0, 2.51, 0.01)
RF = 0.05  # Risk-free rate
RM= 0.0624

eR = RF + betas * (RM-RF)

# Plotting
plt.figure(figsize=(10, 6))

# Plot efficient frontier
plt.plot(betas, eR, label='Security Market Line', color='blue', linewidth=2)

# Plot the Tangent Portfolio (Market Portfolio)
plt.scatter(1, RM, color='black', marker='o', label='Tangent Portfolio (Market Portfolio)')
plt.scatter(0, RF, color='green', marker='o', label='Risk-free portfolio')

arrow_props = dict(facecolor='black', edgecolor='black', shrink=0.05, width=0.1)  # Adjust linewidth for arrows
plt.annotate('Risk-free Asset', [0, RF], [0.1, 0.049], arrowprops=arrow_props)
plt.annotate("Market Portfolio", [1, RM], [1.2, 0.06], arrowprops=arrow_props)


# Adding labels and legend
plt.title("Security Market Line")
plt.xlabel('Beta (Systematic Risk)')
plt.ylabel('Expected Return')
plt.legend()

# Set y-axis and x-axis lower limits to 0
# plt.ylim(bottom=0)
plt.xlim(left=0)
# Display the grid
#plt.grid(True)

# Show the plot
plt.show()
