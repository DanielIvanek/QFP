import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Definujte akcie a váhy
tickers = ['CVS', 'BABA', 'EVO.ST', 'TUI1.DE']  # Akcie v portfoliu
weights = [0.25, 0.25, 0.25, 0.25]  # Váhy akcií (musí dávat dohromady 1)

# 2. Stáhněte historická data
data = yf.download(tickers, start='2015-01-01', end='2023-01-01')['Adj Close']

# 3. Vypočítejte denní výnosy
daily_returns = data.pct_change().dropna()

# 4. Výpočet portfoliového výnosu
portfolio_returns = (daily_returns * weights).sum(axis=1)

# 5. Kumulativní návratnost portfolia
cumulative_returns = (1 + portfolio_returns).cumprod()

# 6. Výkonnostní metriky
annualized_return = cumulative_returns[-1]**(1 / (len(cumulative_returns) / 252)) - 1
annualized_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_volatility

# 7. Zobrazení výsledků
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# 8. Graf kumulativní návratnosti
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Portfolio')
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid()
plt.show()
