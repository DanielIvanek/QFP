import pandas as pd

# tickers in DJIA
# url = 'https://stockmarketmba.com/stocksinthedjia.php'
# tlDJIA = pd.read_html(url)[1]['Symbol'].dropna().tolist()
url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
tlDJIA = pd.read_html(url)[1]['Symbol'].dropna().tolist()

# tickers in SP500
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tlSP500 = pd.read_html(url)[0]['Symbol'].tolist()
