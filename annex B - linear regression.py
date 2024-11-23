# SPLX and ^GSPC

import yfinance as yf
# Download data for SPXL and S&P 500 index
= yf.download('SPXL', start='2020-01-01', end='2023-01-01', auto_adjust = True)
sp500 = yf.download('^GSPC', start='2020-01-01', end='2023-01-01', auto_adjust = True)
# Calculate returns from Adjusted Close prices
spxl_ret = spxl['Close'].pct_change().dropna()
sp500_ret = sp500['Close'].pct_change().dropna()


import numpy as np
# Using polyfit for linear regression
slope, intercept = np.polyfit(sp500_ret, spxl_ret, 1)
predicted_values = np.polyval([slope, intercept], sp500_ret)
print([slope, intercept])


from scipy.stats import linregress
# Using linregress for linear regression
slope, intercept, r_value, p_value, std_err = linregress(sp500_ret, spxl_ret)
predicted_values = intercept + slope * sp500_ret
print([slope, intercept, r_value, p_value, std_err])


from sklearn.linear_model import LinearRegression
#Using LinearRegression
reg = LinearRegression().fit(sp500_ret.values.reshape(-1, 1), spxl_ret)
predicted_values = reg.predict(sp500_ret.values.reshape(-1, 1))
print([reg.coef_, reg.intercept_])


import seaborn as sns
import matplotlib.pyplot as plt
# Create the regression plot
sns.regplot(x=sp500_ret, y=spxl_ret)
# Add labels and title
plt.xlabel('S&P 500 Returns')
plt.ylabel('SPXL ETF Returns')
plt.title('Linear Regression of SPXL ETF on S&P 500 Returns')
plt.show() # Display the plot


import statsmodels.api as sm
#Using statsmodel package
X = sm.add_constant(sp500_ret) # First add a constant to regressors
model = sm.OLS(spxl_ret, X).fit() # Fit the model
predicted_values = model.predict(X)
print(model.summary()) # print the detailed summary of the regression results


#Annualized fees 
print(f"Annualized fees are {-100 * np.power(1 + intercept, 250) - 1:.2f}%")




# YINN and ^HSI

import yfinance as yf
import statsmodels.api as sm
# Download data for YINN and ^HSI at once and drop NaN values
data = yf.download(['YINN', '^HSI'], start='2020-01-01', end='2023-01-01', auto_adjust=True).dropna()
# Calculate returns from Adjusted Close prices
yinn_ret = data['Close']['YINN'].pct_change().dropna()
hsi_ret = data['Close']['^HSI'].pct_change().dropna()
# Perform regression using statsmodel package
X = sm.add_constant(hsi_ret)
model = sm.OLS(yinn_ret, X).fit()
print(model.summary())

# Annualized fees
print(f"Annualized fees are {-100 * np.power(1 + intercept, 250) - 1:.2f}%")