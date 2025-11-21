import pandas as pd
import yfinance as yf

REQUIRED_COLS = [
    "account_id", "asset_type", "ticker", "currency", "quantity", "avg_cost"
]

def load_holdings(filepath="data/sample_holdings.csv") -> pd.DataFrame: #use load_holdings() with own CSV so it works
    """
    Read a holding CSV and validate required columns. It will return a nice DataFrame you can display/use elsewhere.
    """
    df = pd.read_csv(filepath) #read the CSV file into a DataFrame
    missing = [c for c in REQUIRED_COLS if c not in df.columns] #check if any column is missing
    if missing:
        raise ValueError(f"Missing columns in holdings CSV: {missing}")
    df["ticker"] = df["ticker"].astype(str) #put this column in str
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce") #convert into numeric
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")

    if df["quantity"].isna().any() or df["avg_cost"].isna().any():
        raise ValueError("Found non-numeric values in 'quantity' or 'avg_cost'.") #raise an error if not a numeric value in column
    
    return df

def fetch_prices(tickers, start="2020-01-01", end=None) -> pd.DataFrame:
    """
    This function download prices from Yahoo Finance. 
    It also return a DatFrame indexed by date and column.
    
    """
    if isinstance(tickers, (pd.Series, pd.Index)): #if tickers are Serie or Index, turn them into a list
        tickers = tickers.tolist()
    tickers = [str(t).strip() for t in tickers if str(t).strip()] #convert to string and remove spaces or blank
    if not tickers:
        raise ValueError("No tickers provided to fetch_prices().")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    #data.columns = data.columns.droplevel(1)

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns: #normalize yfinance output
        data = data["Adj Close"]
    if isinstance(data, pd.Series):
        colname = tickers[0] if tickers else "price"
        data = data.to_frame(name=colname)

    data = data.rename_axis("date").sort_index()

    # If everything failed (e.g., no internet), generate false price so it doesn't crash
    if data.empty or data.dropna(axis=1, how="all").shape[1] == 0:
        import numpy as np
        print("⚠️ Yahoo download failed; generating synthetic prices for offline use.")
        dates = pd.date_range(start, end or pd.Timestamp.today(), freq="B")
        rng = np.random.default_rng(0)
        fake = pd.DataFrame(index=dates)
        for t in tickers:
            steps = rng.normal(0.0005, 0.02, len(dates))
            fake[t] = 100 * np.exp(np.cumsum(steps))
        fake.index.name = "date"
        return fake

    # Drop all-NaN columns (bad tickers) and return
    data = data.dropna(axis=1, how="all")
    return data

def get_price(ticker:str):
    ticker = ticker.upper() #Be sure that the ticker is all uppercase
    prices = yf.download(ticker, period="max", interval="1mo")
    print(prices.head())
    #removing the double column header 
    prices.columns = prices.columns.droplevel(1)
    print(prices.columns)
    
    adj_close_series = pd.Series(prices["Close"])
    print( adj_close_series.head())
 
if __name__ == "__main__":
    get_price("BYND")
    print(load_holdings())
    pass