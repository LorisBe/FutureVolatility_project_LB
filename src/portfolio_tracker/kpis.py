import numpy as np
import pandas as pd

def annualize_vol(daily_ret: pd.Series) -> float:
    return float(np.sqrt(252) * daily_ret.std(ddof=0))

def sharpe(daily_ret: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily_ret - rf_daily
    denom = excess.std(ddof=0)
    return float(np.sqrt(252) * excess.mean() / (denom + 1e-12))

def drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak

def kpi_table(port_ret: pd.Series) -> pd.DataFrame:
    eq = (1 + port_ret).cumprod()
    dd = drawdown(eq)
    return pd.DataFrame({
        "Cumulative Return": [eq.iloc[-1] - 1],
        "Ann. Vol": [annualize_vol(port_ret)],
        "Sharpe": [sharpe(port_ret)],
        "Max Drawdown": [dd.min()],
    })
