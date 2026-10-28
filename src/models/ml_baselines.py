"""
Classical Machine Learning baselines for RUL prediction:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
"""
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.eval.metrics import compute_all_metrics
from src.utils.config import RANDOM_SEED


def get_ml_models() -> Dict[str, Any]:
    """Returns initialized dictionary of ML baseline models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def train_and_eval_ml_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, pd.Series]]:
    """
    Fits and evaluates Linear Regression, Random Forest, and XGBoost.
    
    Returns:
        metrics_df: summary table of train/val/test metrics for each model
        trained_models: dict of fitted model instances
        importances: dict of feature importance series for RF and XGBoost
    """
    models = get_ml_models()
    records = []
    trained_models = {}
    importances = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        val_metrics = compute_all_metrics(y_val, y_val_pred)
        test_metrics = compute_all_metrics(y_test, y_test_pred)
        
        records.append({
            "Model": name,
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "Val_NASA_Score": val_metrics["NASA_Score"],
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
            "Test_NASA_Score": test_metrics["NASA_Score"],
        })
        
        # Feature importances
        if feature_names is not None:
            if hasattr(model, "feature_importances_"):
                importances[name] = pd.Series(
                    model.feature_importances_, index=feature_names
                ).sort_values(ascending=False)
            elif hasattr(model, "coef_"):
                importances[name] = pd.Series(
                    np.abs(model.coef_), index=feature_names
                ).sort_values(ascending=False)
                
    metrics_df = pd.DataFrame(records)
    return metrics_df, trained_models, importances
