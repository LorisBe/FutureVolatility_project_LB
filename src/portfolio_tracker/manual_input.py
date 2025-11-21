import pandas as pd
from typing import List
import yfinance as yf


def ask_position() -> dict | None:
    """
    This function as the user for his porfolio's position. (ticker, quantity, avg_cost, etc..)
    Return a dict for the position, or None if the user wants to stop
    """" 
    ticker = input("Ticker (empty to finish): ").strip #ask him for the ticker
    if ticker =="":
        return None
    
    qty_str = input("  Quantity: ").strip()
    cost_str = input("  Average cost per unit: ").strip()
    asset_type = input("  Asset type [default: Equity]: ").strip() or "Equity" #ask about all the detail of the position
    currency = input("  Currency [default: USD]: ").strip() or "USD"

    try:
        quantity = float(qty_str)
        avg_cost = float(cost_str) #See if cost and quantity are float, else re do it 
    except ValueError:
        print("  ❌ Quantity and average cost must be numbers. Try again.\n")
        return ask_position()

    pos = {
        "account_id": "acc1",          # simple default
        "asset_type": asset_type,
        "ticker": ticker.upper(),
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
