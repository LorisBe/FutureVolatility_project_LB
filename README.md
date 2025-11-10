# Portfolio Tracker — Data Science Capstone Project

My project builds a data analysis and visualization tool to consolidate investment portfolios. It is useful to see the user's current portfolio return and historical risk. 
The project also uses ML and DS to predict and estimate future risk of the same porfolio.

### Features
- Import holdings from CSV
- Fetch or simulate market prices
- Compute daily returns, portfolio performance, and KPIs
- (Offline-safe: falls back to synthetic prices)
- Interactive Streamlit dashboard

### To run
```bash
pip install -r requirements.txt
streamlit run src/app/dashboard.py
