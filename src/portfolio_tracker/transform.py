import pandas as pd
import numpy as np

def daily_returns(price_wide: pd.DataFrame) -> pd.DataFrame:
    """Percent change of adjusted close prices by column (ticker)."""
    return price_wide.sort_index().pct_change().dropna(how="all")

def portfolio_returns(holdings_df: pd.DataFrame, price_wide: pd.DataFrame) -> pd.Series:
    """
    Value-weighted portfolio daily returns.
    Weights computed once from quantities × first available prices.
    """
    prices = price_wide.sort_index().ffill()
    first = prices.iloc[0]
    shares = holdings_df.set_index("ticker")["quantity"].reindex(prices.columns).fillna(0.0)
    values0 = shares * first
    total0 = float(values0.sum())
    if total0 <= 0:
        raise ValueError("Initial portfolio value is zero—check quantities and tickers.")
    weights = values0 / total0
    rets = daily_returns(prices).fillna(0.0)
    port_ret = (rets * weights).sum(axis=1)
    port_ret.name = "portfolio"
    return port_ret
