# Portfolio Tracker — Data Science Capstone Project

My project builds a data analysis and visualization tool to consolidate investment portfolios. It is useful to see the user's current portfolio return and historical risk. 
The project also uses ML and DS to predict and estimate future risk of the same porfolio.

### Features
- Import holdings from CSV or manually.
- Fetch or simulate market prices via Yahoo Finance
- Price are automatically saved in the data/ folder to see prices
- Compute daily returns, portfolio performance, and KPIs
- Automatique generation od financial plots saved n results/
- Volatility forecasting dataset builder 
- ML forecasting module using 3 models
    1) Naive (baseline)
    2) Linear Regression
    3) Random Forest
- Automatic model selection based on lowest MAE
- Next week volatility prediction
- Portfolio value imnpact estimate based on next 5 days volatility


## How to run the project

## 1) Clone the repository, install requirment and run main()
```bash
git clone <REPO_URL>
cd capstone_project_LB
conda env create -f environment.yml
conda activate capstone_portfolio
python main.py
```
### 2) Enter Loris' base portfolio for same result (adjusted for recent move  in price

Full brake down of my portfolio for you to enter in the code to run and get the result of my personal portfolio : 

**1)**  
Ticker: XRP-USD  
Quantity: 61.722937  
Average cost per unit: 2.2  
Asset type: Crypto  
Currency: USD  

**2)**  
Ticker: BTC-USD  
Quantity: 0.000681  
Average cost per unit: 83682  
Asset type: Crypto  
Currency: USD  

**3)**  
Ticker: NVDA  
Quantity: 2.619475  
Average cost per unit: 175  
Asset type: Equity  
Currency: USD  

**4)**  
Ticker: SPY  
Quantity: 17.481  
Average cost per unit: 523  
Asset type: Equity  
Currency: USD  

**5)**  
Press Enter on an empty ticker to end the input.


# Enjoy!
