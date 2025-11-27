#!/usr/bin/env python

"""
Entry point for the portfolio tracker project.

Run with:
    python main.py

It will:
1. Load sample holdings from data/sample_holdings.csv
2. Fetch prices (yfinance, or synthetic fallback inside fetch_prices)
3. Compute portfolio returns and KPIs
4. Build ML dataset and train Naive / Linear / Random Forest
5. Print model comparison and next-week risk forecast
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.portfolio_tracker.io import load_holdings, fetch_prices
from src.portfolio_tracker.transform import portfolio_returns
from src.portfolio_tracker.kpis import kpi_table, drawdown
from src.portfolio_tracker.risk_models import build_vol_dataset
from src.portfolio_tracker.model_training import train_and_evaluate_all
from src.portfolio_tracker.manual_input import ask_holdings, fetch_prices_from_holdings


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    # === Manual input mode ===
    print("Manual input mode activated. Please enter your positions:\n")
    holdings = ask_holdings()
    print("\nHoldings entered:")
    print(holdings)


    # 2) Fetch prices
    tickers = holdings["ticker"].astype(str)
    print(f"\nFetching prices for tickers: {list(tickers)}")
    prices = fetch_prices_from_holdings(holdings, start="2020-01-01")
    source = prices.attrs.get("source", "Yahoo")
    print(f"Price source: {source}")
    print("\nPrices (head):")
    print(prices.head)

    # 3) Portfolio returns, equity, KPIs
    port_ret = portfolio_returns(holdings, prices)
    equity = (1 + port_ret).cumprod()
    kpis = kpi_table(port_ret)

    print("\n=== Portfolio KPIs ===")
    print(kpis.to_string(float_format=lambda x: f"{x:0.4f}"))

    # Save equity and drawdown plots
    eq_path = results_dir / "equity_curve.png"
    dd_path = results_dir / "drawdown.png"

    plt.figure(figsize=(9, 4))
    equity.plot()
    plt.title("Equity Curve (growth of 1)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(eq_path)
    plt.close()

    dd_series = drawdown(equity)
    plt.figure(figsize=(9, 3))
    dd_series.plot()
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(dd_path)
    plt.close()

    print(f"\nSaved plots to:\n- {eq_path}\n- {dd_path}")

    # 4) Build ML dataset and train models
    X, y = build_vol_dataset(port_ret)
    results, models = train_and_evaluate_all(X, y, train_frac=0.8)

    print("\n=== Volatility Forecast Models ===")
    print(results.to_string(float_format=lambda x: f"{x:0.6f}"))
    first_err_col = results.columns[0] #look for best model
    best_model_name = results[first_err_col].idxmin() 

    print(f"\nBest model according to {first_err_col}: {best_model_name}") #print actual best model to intel the user
    latest_features = X.iloc[[-1]]   # last row of data , keep as DataFrame

    print("\nLatest feature values used for prediction (last available date):")
    print(latest_features.T)  # transpose for easier reading (one feature per line)



    # 5) Predict next-week volatility using Random Forest
    rf_model = models["random_forest"]
    latest_features = X.iloc[[-1]]
    next_vol = float(rf_model.predict(latest_features)[0])
    print(f"\nPredicted next-week annualized volatility: {next_vol:0.4f}")
        
    TRADING_DAYS_PER_YEAR = 252 #will be used tocompute next week vol
    horizon_days = 5

    sigma_5d = next_vol * (horizon_days / TRADING_DAYS_PER_YEAR) ** 0.5
    sigma_5d_pct = sigma_5d * 100
    tickers_list = holdings["ticker"].astype(str).tolist() #current portfolio value (using latest prices) and holdings
    latest_prices = prices.loc[prices.index.max(), tickers_list]
    qty = holdings.set_index("ticker").loc[tickers_list, "quantity"]
    current_value = float((qty.values * latest_prices.values).sum()) #compute actual value of portfolio

    value_up = current_value * (1 + sigma_5d) #estimate if goes upp
    value_down = current_value * (1 - sigma_5d) #same but goes down

    print(f"\nApproximate 5-day volatility (1σ): {sigma_5d_pct:0.2f}%")
    print(
        "Based on this, over the next 5 trading days the portfolio value could typically\n"
        f"move about ±{sigma_5d_pct:0.2f}% around its current value.\n"
        f"Current value ≈ {current_value:0.2f}\n"
        f"Value could range between ≈ [{value_down:0.2f}, {value_up:0.2f}]"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
