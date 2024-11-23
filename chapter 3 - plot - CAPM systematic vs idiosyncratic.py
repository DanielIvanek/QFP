import numpy as np
import matplotlib.pyplot as plt

# Generate an array representing the number of stocks in the portfolio
nstocks = np.arange(5, 60, 1)

# Define a function to represent the idiosyncratic risk
def fce(x):
    return 1 / (((x + 20) ** 3)) * 10e3 / 3 + 0.07

# Calculate idiosyncratic risk, systematic risk, and total risk using the defined function
idiosyncratic_risk = fce(nstocks) - fce(100)
systematic_risk = fce(10e20)
total_risk = idiosyncratic_risk + systematic_risk

# Plotting
plt.figure(figsize=(10, 6))

# Plot a horizontal dashed line for systematic risk
plt.axhline(systematic_risk, color='red', linestyle='--', label='Systematic Risk', linewidth=3)

# Plot the total risk curve
plt.plot(nstocks, total_risk, label='Total Risk', linewidth=3)

# Plot arrows to indicate idiosyncratic risk
arrow_start = (nstocks[5], total_risk[5])  # Adjust index as needed
arrow_end = (nstocks[5], systematic_risk)  # Adjust index as needed
arrow_props = dict(facecolor='black', edgecolor='black', shrink=0.05, width=1)  # Adjust linewidth for arrows
plt.annotate("", arrow_end, (arrow_start[0],arrow_start[1]-0.025), arrowprops=arrow_props)
plt.annotate("", arrow_start, (arrow_end[0],arrow_end[1]+0.025), arrowprops=arrow_props)
plt.text(arrow_start[0]-2, arrow_end[1]+0.02 , ' Idiosyncratic Risk', fontsize=10, rotation='vertical')

# Set y-axis lower limit to 0
plt.ylim(bottom=0)

# Adding labels and legend
plt.title('Portfolio Risk vs. Number of Stocks in Portfolio')
plt.xlabel('Number of Stocks in Portfolio')
plt.ylabel('Portfolio Risk')
plt.legend()

# Display the grid
#plt.grid(True)

# Show the plot
plt.show()
