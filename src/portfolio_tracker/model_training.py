from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train/test sets while preserving time order
    """
    n = len(X) #number of observation
    split_idx = int(n * train_frac)

    X_train = X.iloc[:split_idx].copy() #first 80% train
    X_test = X.iloc[split_idx:].copy() #last 20% test
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def naive_predict(X_test: pd.DataFrame, col_name: str = "rv_5d") -> pd.Series:
    """
    Naive forecast: predict next volatility = current `rv_5d` value.
    """
    if col_name not in X_test.columns:
        raise KeyError(f"Naive feature column '{col_name}' not found in X_test")
    return X_test[col_name] #simply give me the naive volatility of last 5 days


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute basic regression metrics: MAE, MSE, RMSE.
    """
    err = y_true - y_pred #prediction error for each day.
    mae = float(err.abs().mean()) #average absolute error
    mse = float((err ** 2).mean()) #average squared error
    rmse = float(np.sqrt(mse)) #error in same units as volatility.
    return {"Average Absolute Error": mae, "Average Squared Error": mse, "RMSE": rmse}


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """
    Train a simple Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model 


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 5,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a Random Forest regression model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model #train a Random Forest with 300 shallow trees (depth 5), which is enough to capture non-linear patterns but keep it stable


def train_and_evaluate_all(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Split data, train Naive / Linear / Random Forest, and return
    a metrics table plus the trained models.
    """
    X_train, X_test, y_train, y_test = time_series_split(X, y, train_frac=train_frac)

    
    y_pred_naive = naive_predict(X_test, col_name="rv_5d") #Naive model
    naive_metrics = regression_metrics(y_test, y_pred_naive)

    
    lin = train_linear_regression(X_train, y_train) #linear regression model
    y_pred_lin = pd.Series(lin.predict(X_test), index=y_test.index)
    lin_metrics = regression_metrics(y_test, y_pred_lin)

    
    rf = train_random_forest(X_train, y_train) #random forest model
    y_pred_rf = pd.Series(rf.predict(X_test), index=y_test.index)
    rf_metrics = regression_metrics(y_test, y_pred_rf)

    
    results = pd.DataFrame( #Build a single DataFrame with rows = models, columns = MAE/MSE/RMSE
        [naive_metrics, lin_metrics, rf_metrics],
        index=["Naive", "LinearReg", "RandomForest"],
    )

    models = {
        "naive": None,              # baseline has no fitted object
        "linear": lin,
        "random_forest": rf,
    }

    return results, models
