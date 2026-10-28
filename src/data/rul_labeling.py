"""
RUL (Remaining Useful Life) target generation.
Implements both raw linear RUL and standard piecewise-linear RUL (Heimes 2008).
"""
from typing import Optional
import pandas as pd
from src.utils.config import RUL_CAP


def add_rul_labels(
    df: pd.DataFrame, rul_cap: Optional[int] = RUL_CAP, piecewise: bool = True
) -> pd.DataFrame:
    """
    Computes RUL for run-to-failure training data.
    
    In run-to-failure data, each engine runs until failure.
    Raw RUL: max_cycle - current_cycle
    Piecewise linear RUL: min(max_cycle - current_cycle, rul_cap)
    
    Args:
        df: DataFrame containing 'unit_number' and 'time_in_cycles'.
        rul_cap: Maximum RUL ceiling (e.g., 125 cycles).
        piecewise: Whether to cap early-life RUL at rul_cap.
        
    Returns:
        DataFrame with added 'raw_rul' and 'RUL' columns.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["raw_rul"] = max_cycles - df["time_in_cycles"]
    
    if piecewise and (rul_cap is not None):
        df["RUL"] = df["raw_rul"].clip(upper=rul_cap)
    else:
        df["RUL"] = df["raw_rul"]
        
    return df


def add_test_rul_labels(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    rul_cap: Optional[int] = RUL_CAP,
    piecewise: bool = True,
) -> pd.DataFrame:
    """
    Computes ground truth RUL for truncated test trajectories.
    
    For a test engine truncated at T_max, with true final RUL = R_true:
    At any cycle t: raw_rul(t) = (T_max - t) + R_true
    At the last cycle (t = T_max): raw_rul(T_max) = R_true
    
    Args:
        test_df: DataFrame containing test trajectories.
        rul_df: DataFrame with ['unit_number', 'true_rul'].
        rul_cap: Upper bound for piecewise clipping.
        piecewise: Whether to clip target RUL at rul_cap.
        
    Returns:
        test_df augmented with 'raw_rul' and 'RUL' columns.
    """
    test_df = test_df.copy()
    max_test_cycles = test_df.groupby("unit_number")["time_in_cycles"].transform("max")
    
    # Merge true final RUL
    test_df = test_df.merge(rul_df[["unit_number", "true_rul"]], on="unit_number", how="left")
    
    # RUL at each cycle t: (T_max - t) + true_rul
    test_df["raw_rul"] = (max_test_cycles - test_df["time_in_cycles"]) + test_df["true_rul"]
    
    if piecewise and (rul_cap is not None):
        test_df["RUL"] = test_df["raw_rul"].clip(upper=rul_cap)
    else:
        test_df["RUL"] = test_df["raw_rul"]
        
    return test_df
