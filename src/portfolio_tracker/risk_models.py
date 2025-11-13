import numpy as np
import pandas as pd


def realized_vol(daily_ret: pd.Series, window: int = 5, annualized: bool = True) -> pd.Series
    """
    Rolling realized volatility over a given window of specified length
    
    Parameters 
    ----------
    daily_ret : pd.Series
        Daily portfolio returns.
    window : int
        number of days in the volatility window
    annualize : bool
        If true, scale by swrt(252) to annualize.
    
    
    Returns a pd.Series -> Rolling volatility series (aligned with the end of each window)
    
    """

    vol = daily_ret.rolling(window).std(ddof=0) #this function basically takes a window of return from the serie created by daily ret, and compute the volatility for it
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
