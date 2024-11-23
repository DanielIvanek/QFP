import yfinance as yf
import pandas as pd
import riskfolio as rp

start_date = '2012-01-01'
end_date = '2017-01-01'
 
# Download the list of DJIA tickers
url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
ticker_list = pd.read_html(url)[1]['Symbol'].dropna().tolist()
 
# Download historical stock data using the yfinance library.
# We will only keep the closing prices for our analysis.
prices = yf.download(tickers=ticker_list, start=start_date, end=end_date, interval='1d', group_by='column', auto_adjust=True)["Close"]
prices.dropna(axis=1, inplace=True) # Remove any columns with missing values to maintain data consistency
returns=prices.pct_change().dropna() # Calculate returns

port = rp.Portfolio(returns=returns) # Building the portfolio object
port.assets_stats(method_mu='hist', method_cov='hist') # set the method fro mu and covar calculation
port.alpha = 0.05 # set CVaR alpha to 5%

# the cardinality constraint (maximum number of the assets)
#port.card = 5

w = pd.DataFrame()
w['minV'] = port.optimization(model='Classic', rm='MV', obj='MinRisk') # minimum-varinace portfolio
w['maxR'] = port.optimization(model='Classic', rm='MV', obj='MaxRet') # minimum-varinace portfolio
w['minMDD'] = port.optimization(model='Classic', rm='MDD', obj='MinRisk') # minimum-MDD portofolio
w['tanV'] = port.optimization(model='Classic', rm='CVaR', obj='Sharpe', rf=0, hist=True) # Tangential portfolio of mean-variance efficient frontier
w['tanCVaR'] = port.optimization(model='Classic', rm='CVaR', obj='Sharpe', rf=0, hist=True) # Tangential portfolio of mean-CVaR efficient frontier

w['maxU'] = port.optimization(model='Classic', rm='CVaR', obj='Utility', l=2) # maximum-utility portfolio

w['parV'] = port.rp_optimization(model='Classic', rm='MV') # risk parity for variance
w['parCVaR'] = port.rp_optimization(model='Classic', rm='CVaR') # risk parity for CVaR

ax = rp.plot_pie(w=w['minV'].to_frame(), title='Min-CVaR', others=0.05, nrow=25, cmap = "tab20", height=6, width=10, ax=None)

ax = rp.plot_pie(w=w['tanCVaR'].to_frame(), title='Tangential mean-CVaR', others=0.05, nrow=25, cmap = "tab20", height=6, width=10, ax=None)


#-------------------------------------------------------------------------------

port = rp.Portfolio(returns=returns) # Building the portfolio object
port.assets_stats(method_mu='hist', method_cov='hist')
port.alpha = 0.05 # set CVaR alpha to 5%

n_portfolios = 50  

weights = port.efficient_frontier(model='Classic', rm='CVaR', points=n_portfolios, rf=0, hist=True)

# Plot efficient frontier
ax = rp.plot_frontier(w_frontier=weights, mu=port.mu, cov=port.cov, returns=port.returns, rm='CVaR', rf=0, alpha=0.05, cmap='viridis', w=w['tanCVaR'], label='Max Risk-Adjusted Return Portfolio', marker='*', s=16, c='r', height=6, width=10, ax=None)

# Plot efficient frontier portfolio weights
ax = rp.plot_frontier_area(w_frontier=weights, cmap="tab20", height=6, width=10, ax=None)

