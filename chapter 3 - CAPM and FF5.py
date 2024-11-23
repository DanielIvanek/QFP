# -----------------------------------------------------------------------------------------------------------
# CAPM model - simple regresion 
# we assume risk-free rate is 0
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# download the data for Intel and ^GSPC to mimic S&P 500 index and calculate simple returns
data = yf.download(tickers = "INTC ^GSPC", start="2000-11-01", end="2023-10-01", interval = "1d", group_by = 'column', auto_adjust = True)
ret=data["Close"].pct_change().dropna() # calculate simple returns

# plot the returns in scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x=ret['^GSPC'], y=ret['INTC'], s=2)
plt.xlabel('S&P 500 return')
plt.ylabel('INTC return')
plt.title('Dependence of returns')
plt.grid(True)
plt.show()


# plot using seaborn package
sns.jointplot(data=ret, x="^GSPC", y="INTC") # plot the scatter plot with marginal distributions
# sns.jointplot(data=ret, x="^GSPC", y="INTC", kind="kde") # the same but contour plot
sns.regplot(x=ret['^GSPC'], y=ret['INTC'], ci=None) # plot the scatter plot with the regression

# calculate the linear regression using numpy
slope, intercept = np.polyfit(ret['^GSPC'], ret['INTC'], 1)
print(f"intercept: {intercept}; slope: {slope}")

# calculate linear regression with scikit-learn
from sklearn.linear_model import LinearRegression
x = ret['^GSPC'].to_numpy().reshape(-1, 1)
y = ret['INTC'].to_numpy()
model = LinearRegression().fit(x, y)
print(f"intercept: {model.intercept_}; slope: {model.coef_}")

# calculate linear regression with statsmodels
import statsmodels.api as sm

x = ret['^GSPC'].to_numpy()#.reshape(-1, 1)
y = ret['INTC'].to_numpy()
x = sm.add_constant(x) # we must add constant in order to have intercept
model = sm.OLS(y, x) # create the model by means of OLS (be carefull about the order of parameters - different to scikit-learn)
results = model.fit() # fit the model
print(f"intercept: {results.params[0]}; slope: {results.params[1]}")
print(results.summary()) # print results of linear regression

# re-estimate the model without the intercept
x = ret['^GSPC'].to_numpy()
y = ret['INTC'].to_numpy()
model = sm.OLS(y, x)  # create the model by means of OLS 
results = model.fit()  # fit the model
print(results.summary()) # print results of linear regression



# ------------------------------------------------------------------------------------------------------------
# Fama-French 3/4/5 factor models
# data can be downloaded at https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# info about the model can be found at: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_5_factors_2x3.html
import pandas as pd
import statsmodels.api as sm
import urllib.request 
import zipfile
import yfinance as yf

# first we get the data
FF5F_address = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'

urllib.request.urlretrieve(FF5F_address,'FF5F-data.zip') # download the file and save it as 'FF5F-data.zip'
zip_file = zipfile.ZipFile('FF5F-data.zip', 'r') #  open the zip file, we have downloaded
zip_file.extractall()  # extract the data (the extracted file has name 'F-F_Research_Data_5_Factors_2x3_daily.csv')
zip_file.close() # close zip file

ff_factors = pd.read_csv('F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows = 3, index_col = 0)
ff_factors.index = pd.to_datetime(ff_factors.index, format= '%Y%m%d') # format the index
ff_factors = ff_factors.apply(lambda x: x/ 100) # convert the percentage to decimals!

# download the stock data and calculate the returns
data = yf.download(tickers = "INTC", start="2000-11-01", end="2022-11-01", interval = "1d", group_by = 'column', auto_adjust = True)
ret=data["Close"].pct_change().dropna() # calculate simple returns
ret.index = ret.index.tz_localize(None) # remove tz information
# Merging the data
all_data = pd.merge(pd.DataFrame(ret),ff_factors, how = 'inner', left_index= True, right_index= True) # merge data together

all_data.rename(columns={"Close":"R"}, inplace=True) # Rename the column with returns from Close to R 
all_data['R-RF'] = all_data['R'] - all_data['RF'] # Calculate the excess returns

# do the regression wit statsmodels
y = all_data['R-RF']
x = all_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
x = sm.add_constant(x) 
model = sm.OLS(y, x) 
results = model.fit() 
print(results.summary())

# re-estimate the model with only significant variables
y = all_data['R-RF']
x = all_data[['Mkt-RF', 'SMB', 'HML', 'RMW']]
model = sm.OLS(y, x) 
results = model.fit() 
print(results.summary())


# another way 
"""
all_data.rename(columns={'Mkt-RF':'MktRF'}, inplace=True)
model = sm.formula.ols(formula = "R-RF ~ MktRF + SMB + HML + RMW + CMA", data = all_data).fit()
print(model.params)
"""

"""
ideas for the project:
    - for given stoct dataset calculate CAPM beta from daily, monthly, weekly, yearly data and compare
    - for given stoct dataset calculate CAPM beta on rolling window basis, i.e. how the selection of the period influence the value of beta
    - for given stoct dataset calculate beta from CAPM model when the Mrk-RF is negative and when it is possitive and compare
    - find the exact set-up for the imput data to get the beta provided by finance.yahoo.com (monthly returns for 5Y?)
"""

# ---------------------------------------------------------------------------------
#beware that you can obtain the beta from yfinance as follows:
import yfinance as yf
ticker = yf.Ticker('INTC')
stock_info = ticker.info
stock_info['beta']
