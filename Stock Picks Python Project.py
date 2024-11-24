import yfinance as yf
import pandas as pd
import numpy as np

# Funkce pro získání dat
# Aktualizovaný seznam tickerů podle nového seznamu
tickers = [
    'BABA', 'CVS', 'VOW3.DE', '2318.HK', 'BAYN.DE', 'WBD',
    'CEZ.PR', 'EEFT', 'SOFI', 'DIDY', 'BIDU', 'RYAAY',
    'P911.DE', 'HAL', 'PFE', 'EVO.ST', 'TUI1.DE'
]

def get_stock_data(tickers):
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_info = stock.info

            # Získání historických dat pro výpočet volatility a likvidity
            hist_data = stock.history(period="6mo")
            if hist_data.empty:
                print(f"$ {ticker}: No historical data found.")
                continue

            daily_returns = hist_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252)  # Annualizovaná volatilita
            avg_volume = hist_data['Volume'].mean()  # Průměrný denní objem obchodů

            # Výpočet PEG a EPS Growth
            trailing_pe = stock_info.get("trailingPE")
            eps_growth = stock_info.get("earningsGrowth")
            peg_ratio = trailing_pe / eps_growth if trailing_pe and eps_growth else None

            data.append({
                "Ticker": ticker,
                "P/E": trailing_pe,
                "P/B": stock_info.get("priceToBook"),
                "ROE": stock_info.get("returnOnEquity"),
                "EV/EBITDA": stock_info.get("enterpriseToEbitda"),
                "Debt/Equity": stock_info.get("debtToEquity"),
                "Dividend Yield": stock_info.get("dividendYield"),
                "EPS Growth": eps_growth,
                "PEG": peg_ratio,
                "Volatility": volatility,
                "Liquidity": avg_volume
            })
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue
    return pd.DataFrame(data)

# Stažení dat
stock_data = get_stock_data(tickers)

# Zobrazení dat
print("Finanční data akcií:")
print(stock_data)

# Normalizace hodnot
stock_data['PEG_Score'] = 1 / (stock_data['PEG'] + 1).replace([np.inf, -np.inf], np.nan).fillna(0)  # Nižší PEG = vyšší skóre
stock_data['ROE_Score'] = stock_data['ROE'].fillna(0)  # Vyšší ROE = lepší
stock_data['Volatility_Score'] = stock_data['Volatility'].fillna(0)  # Vyšší volatilita = lepší pro swing
stock_data['Debt_Score'] = 1 / (stock_data['Debt/Equity'] + 1).replace([np.inf, -np.inf], np.nan).fillna(0)  # Nižší zadlužení = lepší

# Celkové skóre
stock_data['Total_Score'] = (
    0.4 * stock_data['PEG_Score'] +  # Váha pro PEG
    0.3 * stock_data['ROE_Score'] +  # Váha pro ROE
    0.2 * stock_data['Volatility_Score'] +  # Váha pro volatilitu
    0.1 * stock_data['Debt_Score']  # Váha pro zadlužení
)

# Výběr nejlepších akcií
top_stocks = stock_data.nlargest(5, 'Total_Score')  # Top 5 akcií
print("\nTop akcie podle skóre:")
print(top_stocks)