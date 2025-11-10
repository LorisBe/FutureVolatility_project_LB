# src/app/dashboard.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from portfolio_tracker.io import load_holdings, fetch_prices
from portfolio_tracker.transform import daily_returns, portfolio_returns
from portfolio_tracker.kpis import annualize_vol, sharpe, drawdown

st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("📊 Portfolio Tracker (MVP)")

st.markdown(
    "Upload a CSV with columns: "
    "`account_id, asset_type, ticker, currency, quantity, avg_cost` "
    "or use the sample below."
)

# --- Sidebar inputs
st.sidebar.header("Settings")
start = st.sidebar.date_input("Start date", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("End date", pd.to_datetime("today"))

# --- Upload or sample holdings
uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
if uploaded:
    holdings = pd.read_csv(uploaded)
else:
    holdings = pd.DataFrame({
        "account_id": ["acc1","acc1","acc1"],
        "asset_type": ["Equity","ETF","Crypto"],
        "ticker": ["AAPL","SPY","BTC-USD"],
        "currency": ["USD","USD","USD"],
        "quantity": [10,5,0.02],
        "avg_cost": [150,400,35000],
    })

st.subheader("Holdings")
holdings = st.data_editor(holdings, num_rows="dynamic", use_container_width=True)

# --- Run button
if st.button("Compute"):
    tickers = holdings["ticker"].astype(str)
    prices = fetch_prices(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    st.write("**Prices (head)**")
    st.dataframe(prices.head(), use_container_width=True)
    st.caption(f"Source: {prices.attrs.get('source','unknown')}")

    # Returns & portfolio
    rets = daily_returns(prices)
    port_ret = portfolio_returns(holdings, prices)
    equity = (1 + port_ret).cumprod()

    # KPIs
    kpis = pd.DataFrame({
        "Cumulative Return": [equity.iloc[-1] - 1],
        "Ann. Vol": [annualize_vol(port_ret)],
        "Sharpe": [sharpe(port_ret)],
        "Max Drawdown": [drawdown(equity).min()],
    }).T.rename(columns={0: "Value"})
    st.subheader("KPIs")
    st.dataframe(kpis.style.format({"Value": "{:.4f}"}), use_container_width=True)

    # Charts
    st.subheader("Charts")
    st.write("**Equity Curve (growth of 1)**")
    fig1 = plt.figure()
    equity.plot()
    plt.xlabel("Date"); plt.ylabel("Value")
    st.pyplot(fig1)

    st.write("**Drawdown**")
    dd = drawdown(equity)
    fig2 = plt.figure()
    dd.plot()
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    st.pyplot(fig2)

else:
    st.info("Upload (or edit) holdings and click **Compute**.")
