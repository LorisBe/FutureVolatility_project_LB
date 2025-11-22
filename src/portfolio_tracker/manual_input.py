import sys
from pathlib import Path

# Make sure Python sees the project root so "src" is importable
ROOT = Path(__file__).resolve().parents[2]  # -> /files/capstone_project_LB
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pandas as pd
from typing import List, Dict, Optional
import yfinance as yf


from src.portfolio_tracker.io import fetch_prices
from src.portfolio_tracker.transform import portfolio_returns
from src.portfolio_tracker.kpis import kpi_table
from src.portfolio_tracker.risk_models import build_vol_dataset

def ask_position() -> Optional[Dict]:
    """
    This function as the user for his porfolio's position. (ticker, quantity, avg_cost, etc..)
    Return a dict for the position, or None if the user wants to stop
    """
    ticker = input("Ticker (empty to finish): ").strip() #ask him for the ticker
    if ticker == "" :
        return None
    
    qty_str = input("  Quantity: ").strip()
    cost_str = input("  Average cost per unit: ").strip()
    asset_type = input("  Asset type [default: Equity]: ").strip() or "Equity" #ask about all the detail of the position
    currency = input("  Currency [default: USD]: ").strip() or "USD"

    try:
        quantity = float(qty_str)
        avg_cost = float(cost_str) #See if cost and quantity are float, else re do it 
    except ValueError:
        print(" Quantity and average cost must be numbers. Try again.\n")
        return ask_position()

    pos = {
        "account_id": "acc1",          # simple default
        "asset_type": asset_type,
        "ticker": ticker,
        "currency": currency.upper(), 
        "quantity": quantity,
        "avg_cost": avg_cost,
    }
    return pos #return a dictionnary for that single position -> will be looped in future function to create multiple

def ask_holdings() -> pd.DataFrame:
    """
    Loop call of ask_position() until user stops, then build and return a holdings DataFrame
    """
    rows: List[dict] = []
    i = 1
    
    while True:
        print(f"Position #{i}") #use the last function unitl user breaks
        pos = ask_position()
        if pos is None: #user finished his holdings
            break
        rows.append(pos)
        i+= 1
        print()
        
    if not rows:
        raise ValueError("No positions entered")
    
    df = pd.DataFrame(rows)
    print("Holdings caputred, DataFrame of holdings created, please check accruacy:")
    print(df)
    return df

def fetch_prices_from_holdings(holdings, start="2020-01-01", end=None):
    """
    FORCE correct price format: wide [date x tickers] with adjusted close only.
    """
    tickers = holdings["ticker"].tolist()

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # If yfinance returns multi-index OHLCV, select the 'Close' level
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # Ensure correct ordering and fill missing values
    data = data.sort_index().ffill()

    return data

if __name__ == "__main__":
    print("\n=== Manual portfolio test ===\n")

    # 1) Ask user for holdings
    holdings = ask_holdings()

    print("\n=== Holdings DataFrame ===")
    print(holdings)

    # 2) Fetch prices using your existing fetch_prices() via our wrapper
    prices = fetch_prices_from_holdings(holdings, start="2020-01-01")

    print("\n=== Prices DataFrame (head) ===")
    print(prices.head())

    # 3) Compute portfolio daily returns
    port_ret = portfolio_returns(holdings, prices)
    print("\n=== Portfolio daily returns (head) ===")
    print(port_ret.head())

    # 4) Compute KPIs
    kpis = kpi_table(port_ret)
    print("\n=== KPI table ===")
    print(kpis)

    print("\n=== Building ML dataset (X, y) ===")
X, y = build_vol_dataset(port_ret)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nX head:\n", X.head())
print("\ny head:\n", y.head())
