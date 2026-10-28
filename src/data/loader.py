"""
Data loader for NASA C-MAPSS turbofan engine degradation datasets.
"""
import urllib.request
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.utils.config import (
    ALL_COLUMNS,
    CMAPSS_BASE_URL,
    DATA_RAW_DIR,
)


def download_file(filename: str, target_dir: Path = DATA_RAW_DIR) -> Path:
    """Download a single C-MAPSS dataset file if not present locally."""
    target_path = target_dir / filename
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"{CMAPSS_BASE_URL}{filename}"
    print(f"Downloading {filename} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
        f.write(resp.read())
    print(f"Saved to {target_path} ({target_path.stat().st_size} bytes)")
    return target_path


def load_dataset(
    dataset_id: str = "FD001", download_if_missing: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, test, and true RUL data for a specified C-MAPSS sub-dataset (e.g. 'FD001').

    Returns:
        train_df: DataFrame with 26 columns (unit_number, time_in_cycles, 3 settings, 21 sensors)
        test_df: DataFrame with 26 columns
        rul_df: DataFrame containing the ground truth RUL for each test unit
    """
    train_file = f"train_{dataset_id}.txt"
    test_file = f"test_{dataset_id}.txt"
    rul_file = f"RUL_{dataset_id}.txt"

    if download_if_missing:
        download_file(train_file)
        download_file(test_file)
        download_file(rul_file)

    train_path = DATA_RAW_DIR / train_file
    test_path = DATA_RAW_DIR / test_file
    rul_path = DATA_RAW_DIR / rul_file

    if not train_path.exists() or not test_path.exists() or not rul_path.exists():
        raise FileNotFoundError(
            f"Missing one or more files for {dataset_id} in {DATA_RAW_DIR}. "
            f"Please drop {train_file}, {test_file}, and {rul_file} into {DATA_RAW_DIR}."
        )

    # C-MAPSS files are space-delimited with trailing whitespace
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    train_df = train_df.iloc[:, :26]
    train_df.columns = ALL_COLUMNS

    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    test_df = test_df.iloc[:, :26]
    test_df.columns = ALL_COLUMNS

    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_rul"])
    rul_df["unit_number"] = rul_df.index + 1

    # Ensure integer types for identifiers and cycles
    train_df["unit_number"] = train_df["unit_number"].astype(int)
    train_df["time_in_cycles"] = train_df["time_in_cycles"].astype(int)
    test_df["unit_number"] = test_df["unit_number"].astype(int)
    test_df["time_in_cycles"] = test_df["time_in_cycles"].astype(int)
    rul_df["unit_number"] = rul_df["unit_number"].astype(int)

    return train_df, test_df, rul_df


if __name__ == "__main__":
    print("Testing loader on FD001...")
    tr, te, ru = load_dataset("FD001")
    print(f"Train shape: {tr.shape}, Test shape: {te.shape}, RUL shape: {ru.shape}")
    print(f"Unique train units: {tr['unit_number'].nunique()}, Unique test units: {te['unit_number'].nunique()}")
