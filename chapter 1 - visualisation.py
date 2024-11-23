import yfinance as yf
import pandas as pd
 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Download data
tickers = ["META",  # Facebook/Meta Platforms, Inc.
           "AAPL",  # Apple
           "AMZN",  # Amazon
           "NFLX",  # Netflix
           "GOOGL"  # Google (Alphabet Inc.)
          ]
 
data = yf.download(tickers=tickers, start='2013-01-01', end='2023-01-01', interval='1d', group_by='column', auto_adjust=True)
prices = data["Close"].dropna()
simple_returns = prices.pct_change().fillna(0)
v = pd.DataFrame(index=prices.columns, columns=['quantity'], data=[1,2,3,4,5])
P_times_v = prices.multiply(v['quantity'], axis=1)
wealth = P_times_v.sum(1)

# Scatter plot using Matplotlib
plt.figure(figsize=(10,6))
plt.scatter(simple_returns['META'], simple_returns['AAPL'])
plt.title('Scatter plot of Returns: Meta vs. Apple (Matplotlib)')
plt.xlabel('Meta Returns')
plt.ylabel('Apple Returns')
plt.grid(True)
plt.show()
 
# Scatter plot using Seaborn
sns.scatterplot(x=simple_returns['META'], y=simple_returns['AAPL'])
plt.title('Scatter plot of Returns: Meta vs. Apple (Seaborn)')
plt.xlabel('Meta Returns')
plt.ylabel('Apple Returns')
plt.show()
 
# Scatter plot using Pandas
simple_returns.plot.scatter(x='META', y='AAPL', figsize=(10,6), grid=True)
plt.title('Scatter plot of Returns: Meta vs. Apple (Pandas)')
plt.xlabel('Meta Returns')
plt.ylabel('Apple Returns')
plt.show()
 
# Scatter plot using Plotly
fig = px.scatter(simple_returns, x='META', y='AAPL', title='Scatter plot of Returns: Meta vs. Apple (Plotly)')
fig.show()


# Line plot using Matplotlib
plt.figure(figsize=(10,6))
plt.plot(wealth, label='Wealth', color='blue')
plt.title('Wealth Over Time (Matplotlib)')
plt.xlabel('Date')
plt.ylabel('Wealth')
plt.legend()
plt.grid(True)
plt.show()

# Line plot using Seaborn
plt.figure(figsize=(10,6))
sns.lineplot(x=wealth.index, y=wealth.values, label='Wealth')
plt.title('Wealth Over Time (Seaborn)')
plt.xlabel('Date')
plt.ylabel('Wealth')
plt.legend()
plt.grid(True)
plt.show()

# Line plot using Pandas
wealth.plot(figsize=(10,6), label='Wealth', title='Wealth Over Time (Pandas)', grid=True)
plt.xlabel('Date')
plt.ylabel('Wealth')
plt.legend()
plt.show()

# Line plot using Plotly
fig = px.line(wealth, x=wealth.index, y=wealth.values, title='Wealth Over Time (Plotly)', labels={'y':'Wealth'})
fig.show()



# Stacked area plot using Matplotlib
plt.figure(figsize=(10,6))
plt.stackplot(P_times_v.index, [P_times_v[col] for col in P_times_v.columns], labels=P_times_v.columns)
plt.title('Stock Value Over Time (Matplotlib)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Stacked area plot using Pandas
P_times_v.plot.area(figsize=(10,6), title='Stock Value Over Time (Pandas)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


# Stacked area plot using Plotly Express
import plotly.express as px
fig = px.area(P_times_v, x=P_times_v.index, y=P_times_v.columns, 
              title='Stock Value Over Time (Plotly Express)')
fig.show()




# Creating a figure usig plt.subplot with shared x-axis
plt.figure(figsize=(10, 6))

# Creating the first subplot for prices
ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
ax1.plot(prices['META'].iloc[1001:], label='META Prices', color='blue')
ax1.set_title('META Stock Prices')
ax1.set_ylabel('Price')

# Creating the second subplot for simple returns
ax2 = plt.subplot(2, 1, 2, sharex=ax1)  # 2 rows, 1 column, second plot
ax2.plot(simple_returns['META'], label='META Simple Returns', color='red')
ax2.set_title('META Stock Simple Returns')
ax2.set_xlabel('Date')
ax2.set_ylabel('Simple Return')

# Adjusting the layout
plt.tight_layout()
plt.show()


# Creating a figure with two subplots with aligned x-axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plotting prices on the first subplot
ax1.plot(prices['META'].iloc[1001:], label='META Prices', color='blue')
ax1.set_title('META Stock Prices')
ax1.set_ylabel('Price')

# Plotting simple returns on the second subplot
ax2.plot(simple_returns['META'], label='META Simple Returns', color='red')
ax2.set_title('META Stock Simple Returns')
ax2.set_xlabel('Date')
ax2.set_ylabel('Simple Return')

# Adjusting the layout
plt.tight_layout()
plt.show()


