import pandas as pd
import yfinance as yf  # we import the library and rename it to yf
import matplotlib.pyplot as plt
import numpy as np 

ticker = yf.Ticker("AAPL")
 
expirations = ticker.options
print(expirations)
 
c = ticker.option_chain(expirations[5]).calls 
print(c.head())
  
fig = plt.figure(figsize=(10, 6))
plt.plot(c.strike,c.lastPrice, label=expirations[5])
 
all_calls=pd.DataFrame()
for expiration in expirations[6:-2]:
    c = ticker.option_chain(expiration).calls
    plt.plot(c.strike,c.lastPrice, label=expiration)
    c["expiration"]=expiration
    all_calls = pd.concat([all_calls, c], axis=0)

all_calls['expiration'] = pd.to_datetime(all_calls['expiration'])
all_calls['days_to_expiry'] = (all_calls['expiration'] - pd.Timestamp.now()).dt.days
all_calls['midPrice'] = (all_calls['ask']+all_calls['bid'])/2

# Adding legend, x-axis and y-axis labels, and a figure title
plt.legend(loc='upper right')
plt.xlabel('Strike Price')
plt.ylabel('Last Price')
plt.title('Option Prices for AAPL')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# 3D plot via matplotlib
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_calls['days_to_expiry'], all_calls['strike'], all_calls['lastPrice'])
ax.set_xlabel('Days to Expiry')
ax.set_ylabel('Strike Price')
ax.set_zlabel('Option Price')
ax.set_title('Matplotlib: Option Prices vs. Maturity and Strike Price')
plt.show()


import plotly.express as px
fig = px.scatter_3d(all_calls, x='days_to_expiry', y='strike', z='lastPrice',
                    title='Plotly: Option Prices vs. Maturity and Strike Price')
fig.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
# Pivot the data to get a grid suitable for a surface plot
pivot_table = all_calls.pivot(index='days_to_expiry', columns='strike', values='lastPrice')
 
X = pivot_table.columns
Y = pivot_table.index
X, Y = np.meshgrid(X, Y)
Z = pivot_table.values
 
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
 
ax.set_xlabel('Strike Price')
ax.set_ylabel('Days to Expiry')
ax.set_zlabel('Option Price')
ax.set_title('Matplotlib: Option Prices Surface Plot')
plt.show()
 


