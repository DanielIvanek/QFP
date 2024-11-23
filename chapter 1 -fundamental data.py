import yfinance as yf
 
ticker = yf.Ticker('INTC')
 
info = ticker.info # get all the information
CAPMbeta = info['beta'] # get the beta of CAPM model
  
# basic information 
holders = ticker.major_holders # info about holders
holder_inst = ticker.institutional_holders # institutional holders and their share
holder_mf =ticker.mutualfund_holders # mutual funds holders and their share
 
# dividends and splits
actions = ticker.actions # get the dividends and splits
dividennds = ticker.dividends # get only the dividends
splits =  ticker.splits # get only the splits
 
#financials 
stat_inc = ticker.income_stmt # Income statement for last 4 years 
stat_inc_q = ticker.quarterly_income_stmt # Income statement for last 4 quarters
stat_bal = ticker.balance_sheet # Balance sheet for last 4 years 
stat_bal_q = ticker.quarterly_balance_sheet # Balance sheet for last 4 quarters
stat_cf = ticker.cashflow # Cash-flow statement for last 4 years 
stat_cf_q = ticker.quarterly_cashflow # Cash-flow statement for last 4 quarters 
 

news = ticker.news # get related news