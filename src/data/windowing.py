"""
Sliding-window sequence generation for deep sequence models (MLP, LSTM, GRU, CNN-LSTM).
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import SEQUENCE_LENGTH


def generate_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "RUL",
    seq_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs sliding windows of length seq_length from engine trajectories.
    
    Args:
        df: DataFrame containing unit_number, time_in_cycles, features, and target.
        feature_cols: List of sensor/setting column names.
        target_col: Name of target column (e.g., 'RUL').
        seq_length: Window size W (default 30).
        
    Returns:
        X: 3D array of shape (N_windows, seq_length, n_features)
        y: 1D array of shape (N_windows,)
    """
    X_list = []
    y_list = []
    
    for _, group in df.groupby("unit_number"):
        group_sorted = group.sort_values("time_in_cycles")
        data_mat = group_sorted[feature_cols].values
        labels = group_sorted[target_col].values
        n_cycles = len(group_sorted)
        
        if n_cycles < seq_length:
            # Pad at the start by repeating the earliest cycle readings
            pad_len = seq_length - n_cycles
            pad_data = np.repeat(data_mat[:1], pad_len, axis=0)
            padded_data = np.vstack([pad_data, data_mat])
            X_list.append(padded_data)
            y_list.append(labels[-1])
        else:
            for i in range(n_cycles - seq_length + 1):
                X_list.append(data_mat[i : i + seq_length])
                y_list.append(labels[i + seq_length - 1])
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def generate_test_windows(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = SEQUENCE_LENGTH,
    rul_cap: float = 125.0,
    piecewise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the final window of length seq_length for each test engine.
    Matches the test engine against its official ground truth RUL.
    
    Returns:
        X_test: shape (n_test_engines, seq_length, n_features)
        y_test: shape (n_test_engines,) true final RUL (clipped if piecewise=True)
        unit_ids: shape (n_test_engines,) engine unit numbers
    """
    X_test_list = []
    y_test_list = []
    unit_ids = []
    
    rul_dict = dict(zip(rul_df["unit_number"], rul_df["true_rul"]))
    
    for unit, group in test_df.groupby("unit_number"):
        group_sorted = group.sort_values("time_in_cycles")
        data_mat = group_sorted[feature_cols].values
        n_cycles = len(group_sorted)
        
        if n_cycles < seq_length:
            pad_len = seq_length - n_cycles
            pad_data = np.repeat(data_mat[:1], pad_len, axis=0)
            window = np.vstack([pad_data, data_mat])
        else:
            window = data_mat[-seq_length:]
            
        true_rul = float(rul_dict[unit])
        if piecewise and (rul_cap is not None):
            true_rul = min(true_rul, float(rul_cap))
            
        X_test_list.append(window)
        y_test_list.append(true_rul)
        unit_ids.append(unit)
        
    return (
        np.array(X_test_list, dtype=np.float32),
        np.array(y_test_list, dtype=np.float32),
        np.array(unit_ids, dtype=np.int32),
    )


class TurbofanDataset(Dataset):
    """PyTorch Dataset for turbofan sliding windows."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self) -> int:
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
