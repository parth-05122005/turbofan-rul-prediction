"""
Feature engineering and normalization for C-MAPSS sensor time series.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.config import SENSOR_COLUMNS, SETTING_COLUMNS


def identify_flat_sensors(train_df: pd.DataFrame, std_threshold: float = 1e-4) -> List[str]:
    """
    Empirically detects flat or near-zero variance sensors in the training dataset.
    """
    flat = []
    for s in SENSOR_COLUMNS:
        if s in train_df.columns:
            if train_df[s].std() < std_threshold:
                flat.append(s)
    return flat


def get_informative_features(train_df: pd.DataFrame, include_settings: bool = False) -> List[str]:
    """
    Returns the list of informative feature columns (excluding flat sensors).
    """
    flat = identify_flat_sensors(train_df)
    features = [s for s in SENSOR_COLUMNS if s not in flat and s in train_df.columns]
    if include_settings:
        features = [c for c in SETTING_COLUMNS if c in train_df.columns] + features
    return features


def fit_transform_scaler(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "minmax",
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Fits scaler on train features, transforms both train and test data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1)) if scaler_type == "minmax" else StandardScaler()
    
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    scaler.fit(train_df[feature_cols])
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_scaled, test_scaled, scaler


def extract_tabular_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Extracts engineered rolling statistics for classical ML baselines.
    Features:
    - Raw sensor readings (normalized)
    - Normalized cycle count
    - Rolling mean and rolling std for each sensor across multiple window sizes
    - Rolling min and max
    """
    df_out = df.copy()
    grouped = df_out.groupby("unit_number")
    
    # Normalized cycle feature (within engine history)
    df_out["cycle_norm"] = df_out["time_in_cycles"] / 100.0
    
    new_cols = {}
    for w in windows:
        for col in feature_cols:
            mean_series = grouped[col].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            std_series = grouped[col].rolling(w, min_periods=1).std().fillna(0.0).reset_index(level=0, drop=True)
            min_series = grouped[col].rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            max_series = grouped[col].rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            
            new_cols[f"{col}_roll_mean_{w}"] = mean_series
            new_cols[f"{col}_roll_std_{w}"] = std_series
            new_cols[f"{col}_roll_min_{w}"] = min_series
            new_cols[f"{col}_roll_max_{w}"] = max_series
            
    df_engineered = pd.concat([df_out, pd.DataFrame(new_cols, index=df_out.index)], axis=1)
    return df_engineered
