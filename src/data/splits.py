"""
Engine-level train/val splits and federated client partitioning (IID vs Non-IID).
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def train_val_split_by_engine(
    df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame by engine unit_number so an engine's trajectory never leaks across splits.
    """
    unique_units = np.array(sorted(df["unit_number"].unique()))
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_units)
    
    n_val = max(1, int(len(unique_units) * val_ratio))
    val_units = set(unique_units[:n_val])
    train_units = set(unique_units[n_val:])
    
    train_df = df[df["unit_number"].isin(train_units)].copy()
    val_df = df[df["unit_number"].isin(val_units)].copy()
    return train_df, val_df


def partition_engines_iid(
    df: pd.DataFrame, num_clients: int = 3, random_state: int = 42
) -> Dict[int, pd.DataFrame]:
    """
    Partitions engines uniformly at random across num_clients (IID distribution).
    """
    unique_units = np.array(sorted(df["unit_number"].unique()))
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_units)
    
    shards = np.array_split(unique_units, num_clients)
    client_shards = {}
    for cid, units in enumerate(shards):
        client_shards[cid] = df[df["unit_number"].isin(units)].copy()
        
    return client_shards


def partition_engines_non_iid(
    df: pd.DataFrame, num_clients: int = 3, strategy: str = "lifespan"
) -> Dict[int, pd.DataFrame]:
    """
    Partitions engines heterogeneously across num_clients (Non-IID distribution).
    
    Strategy 'lifespan': Sorts engines by max cycle duration (short-lived vs medium vs long-lived).
    Simulates different factory operating profiles and stress regimes.
    """
    if strategy == "lifespan":
        max_cycles = df.groupby("unit_number")["time_in_cycles"].max().sort_values()
        sorted_units = max_cycles.index.values
        shards = np.array_split(sorted_units, num_clients)
        
        client_shards = {}
        for cid, units in enumerate(shards):
            client_shards[cid] = df[df["unit_number"].isin(units)].copy()
        return client_shards
    else:
        raise ValueError(f"Unknown non-IID partitioning strategy: {strategy}")
