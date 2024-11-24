import yfinance as yf
import pandas as pd


# Funkce pro výpočet vnitřní hodnoty na základě reálného P/E a EPS
def pe_eps_real_model(tickers):
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Načtení dat
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            eps = info.get("trailingEps", None)
            pe_ratio = info.get("trailingPE", None)  # Načtení skutečného P/E

            # Ověření dostupnosti dat
            if eps is None or pe_ratio is None:
                print(f"{ticker}: EPS nebo P/E není dostupné.")
                continue

            # Výpočet vnitřní hodnoty
            intrinsic_value = eps * pe_ratio

            # Uložení výsledků
            results.append({
                "Ticker": ticker,
                "Current Price": current_price,
                "EPS": eps,
                "P/E Ratio": pe_ratio,
                "Intrinsic Value": intrinsic_value,
                "Undervalued": current_price < intrinsic_value
            })
        except Exception as e:
            print(f"Chyba při zpracování {ticker}: {e}")
            continue

    return results


# Seznam tickerů
tickers = [
    'BABA', 'CVS', 'VOW3.DE', '2318.HK', 'BAYN.DE', 'WBD',
    'CEZ.PR', 'EEFT', 'SOFI', 'BIDU', 'RYAAY',
    'P911.DE', 'HAL', 'PFE', 'EVO.ST', 'TUI1.DE', 'AAPL'
]

# Výpočet vnitřních hodnot
pe_eps_real_results = pe_eps_real_model(tickers)

# Zobrazení výsledků
pe_eps_real_df = pd.DataFrame(pe_eps_real_results)
print(pe_eps_real_df)

# Uložení výsledků do CSV
pe_eps_real_df.to_csv("pe_eps_real_intrinsic_values.csv", index=False)
