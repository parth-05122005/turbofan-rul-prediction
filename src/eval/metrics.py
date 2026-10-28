"""
Evaluation metrics for Remaining Useful Life (RUL) prediction.
Includes standard regression metrics (MAE, RMSE, R²) and the NASA asymmetric scoring function.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS / PHM 2008 asymmetric scoring function.
    Penalizes late predictions (y_pred > y_true) more heavily than early predictions (y_pred < y_true).
    
    d_i = y_pred_i - y_true_i
    S_i = exp(-d_i / 13) - 1   if d_i < 0  (early prediction)
    S_i = exp( d_i / 10) - 1   if d_i >= 0 (late prediction)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    
    diff = y_pred - y_true
    scores = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics and the NASA asymmetric score.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    nasa_score = float(nasa_scoring_function(y_true, y_pred))
    nasa_mean = float(nasa_score / len(y_true)) if len(y_true) > 0 else 0.0
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NASA_Score": nasa_score,
        "NASA_Mean_Score": nasa_mean,
    }
