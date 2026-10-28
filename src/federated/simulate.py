"""
Simulation runner comparing Centralized vs Local-only vs Federated (IID & Non-IID) training.
"""
from typing import Dict, List, Tuple, Any
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import flwr as fl

from src.utils.config import (
    SEQUENCE_LENGTH,
    RUL_CAP,
    RANDOM_SEED,
    TABLES_DIR,
    FIGURES_DIR,
)
from src.data.loader import load_dataset
from src.data.rul_labeling import add_rul_labels, add_test_rul_labels
from src.features.engineering import get_informative_features, fit_transform_scaler
from src.data.splits import (
    train_val_split_by_engine,
    partition_engines_iid,
    partition_engines_non_iid,
)
from src.data.windowing import (
    generate_sliding_windows,
    generate_test_windows,
    TurbofanDataset,
)
from src.models.cnn_lstm import CNNLSTMRegressor
from src.models.train_utils import train_deep_model, evaluate_model_on_windows
from src.federated.client import TurbofanClient
from src.federated.server import get_fedavg_strategy, create_centralized_eval_fn
from src.eval.metrics import compute_all_metrics


def prepare_data():
    """Loads, scales, and prepares sliding windows for FD001."""
    train_df, test_df, rul_df = load_dataset("FD001")
    feature_cols = get_informative_features(train_df, include_settings=False)
    
    train_df_rul = add_rul_labels(train_df, rul_cap=RUL_CAP, piecewise=True)
    test_df_rul = add_test_rul_labels(test_df, rul_df, rul_cap=RUL_CAP, piecewise=True)
    
    train_scaled, test_scaled, scaler = fit_transform_scaler(
        train_df_rul, test_df_rul, feature_cols, scaler_type="minmax"
    )
    
    X_test, y_test, test_units = generate_test_windows(
        test_scaled, rul_df, feature_cols, seq_length=SEQUENCE_LENGTH, rul_cap=RUL_CAP, piecewise=True
    )
    
    return train_scaled, test_scaled, rul_df, feature_cols, X_test, y_test


def train_centralized(
    train_scaled: pd.DataFrame,
    feature_cols: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
) -> Tuple[CNNLSTMRegressor, Dict[str, float]]:
    """Trains centralized CNN-LSTM upper bound."""
    print("\n--- Training Centralized CNN-LSTM ---")
    train_split, val_split = train_val_split_by_engine(train_scaled, val_ratio=0.2, random_state=RANDOM_SEED)
    
    X_tr, y_tr = generate_sliding_windows(train_split, feature_cols, "RUL", SEQUENCE_LENGTH)
    X_va, y_va = generate_sliding_windows(val_split, feature_cols, "RUL", SEQUENCE_LENGTH)
    
    tr_loader = DataLoader(TurbofanDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    va_loader = DataLoader(TurbofanDataset(X_va, y_va), batch_size=64, shuffle=False)
    
    model = CNNLSTMRegressor(n_features=len(feature_cols))
    model, _ = train_deep_model(model, tr_loader, va_loader, epochs=epochs, lr=1e-3, verbose=False)
    
    _, metrics = evaluate_model_on_windows(model, X_test, y_test)
    print(f"Centralized CNN-LSTM Test Metrics -> MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.3f}")
    return model, metrics


def train_local_only(
    client_shards: Dict[int, pd.DataFrame],
    feature_cols: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Trains isolated local CNN-LSTM models for each factory client."""
    print("\n--- Training Local-Only Models (No Federation) ---")
    local_results = {}
    
    for cid, shard in client_shards.items():
        tr_split, va_split = train_val_split_by_engine(shard, val_ratio=0.2, random_state=RANDOM_SEED)
        X_tr, y_tr = generate_sliding_windows(tr_split, feature_cols, "RUL", SEQUENCE_LENGTH)
        X_va, y_va = generate_sliding_windows(va_split, feature_cols, "RUL", SEQUENCE_LENGTH)
        
        tr_loader = DataLoader(TurbofanDataset(X_tr, y_tr), batch_size=64, shuffle=True)
        va_loader = DataLoader(TurbofanDataset(X_va, y_va), batch_size=64, shuffle=False)
        
        model = CNNLSTMRegressor(n_features=len(feature_cols))
        model, _ = train_deep_model(model, tr_loader, va_loader, epochs=epochs, lr=1e-3, verbose=False)
        
        _, metrics = evaluate_model_on_windows(model, X_test, y_test)
        local_results[f"Client_{cid}"] = metrics
        print(f"Client {cid} Local-Only Test -> MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.3f}")
        
    avg_mae = np.mean([m["MAE"] for m in local_results.values()])
    avg_rmse = np.mean([m["RMSE"] for m in local_results.values()])
    avg_r2 = np.mean([m["R2"] for m in local_results.values()])
    avg_nasa = np.mean([m["NASA_Score"] for m in local_results.values()])
    
    summary = {
        "MAE": float(avg_mae),
        "RMSE": float(avg_rmse),
        "R2": float(avg_r2),
        "NASA_Score": float(avg_nasa),
        "per_client": local_results,
    }
    return summary


def run_federated_simulation(
    client_shards: Dict[int, pd.DataFrame],
    feature_cols: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_rounds: int = 15,
    local_epochs: int = 2,
    sim_name: str = "IID",
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Runs Flower simulation for FedAvg across simulated factory clients."""
    print(f"\n--- Running Federated Learning Simulation ({sim_name}) ---")
    
    # Pre-generate numpy arrays for each client shard to ensure safe Ray actor serialization
    client_arrays = {}
    for cid, shard in client_shards.items():
        tr_split, va_split = train_val_split_by_engine(shard, val_ratio=0.2, random_state=RANDOM_SEED)
        X_tr, y_tr = generate_sliding_windows(tr_split, feature_cols, "RUL", SEQUENCE_LENGTH)
        X_va, y_va = generate_sliding_windows(va_split, feature_cols, "RUL", SEQUENCE_LENGTH)
        client_arrays[str(cid)] = (X_tr, y_tr, X_va, y_va)
        
    test_loader = DataLoader(TurbofanDataset(X_test, y_test), batch_size=64, shuffle=False)
    eval_model_template = CNNLSTMRegressor(n_features=len(feature_cols))
    initial_weights = [val.cpu().numpy() for _, val in eval_model_template.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)
    
    convergence_logs = []
    
    def logging_eval_fn(round_num, params, config):
        eval_fn = create_centralized_eval_fn(eval_model_template, test_loader, device="cpu")
        res = eval_fn(round_num, params, config)
        if res is not None:
            loss, m = res
            convergence_logs.append(m)
        return res

    def client_fn(cid: str) -> fl.client.Client:
        X_tr, y_tr, X_va, y_va = client_arrays[cid]
        tr_loader = DataLoader(TurbofanDataset(X_tr, y_tr), batch_size=64, shuffle=True)
        va_loader = DataLoader(TurbofanDataset(X_va, y_va), batch_size=64, shuffle=False)
        c_model = CNNLSTMRegressor(n_features=len(feature_cols))
        return TurbofanClient(
            cid=cid,
            model=c_model,
            train_loader=tr_loader,
            val_loader=va_loader,
            device="cpu",
            local_epochs=local_epochs,
            lr=1e-3,
        ).to_client()

    strategy = get_fedavg_strategy(
        eval_fn=logging_eval_fn,
        initial_parameters=initial_parameters,
        min_fit_clients=len(client_shards),
        min_available_clients=len(client_shards),
    )

    # Launch simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_shards),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    final_metrics = convergence_logs[-1] if convergence_logs else {}
    return final_metrics, convergence_logs


def run_full_federated_study(num_clients: int = 3, num_rounds: int = 15, local_epochs: int = 2):
    """Executes the complete study and saves comparison tables."""
    train_scaled, test_scaled, rul_df, feature_cols, X_test, y_test = prepare_data()
    
    # 1. Centralized Model
    _, cent_metrics = train_centralized(train_scaled, feature_cols, X_test, y_test, epochs=num_rounds * local_epochs)
    
    # 2. Local-Only Models (IID partition)
    iid_shards = partition_engines_iid(train_scaled, num_clients=num_clients, random_state=RANDOM_SEED)
    local_iid_summary = train_local_only(iid_shards, feature_cols, X_test, y_test, epochs=num_rounds * local_epochs)
    
    # 3. Federated IID
    fed_iid_metrics, iid_curve = run_federated_simulation(
        iid_shards, feature_cols, X_test, y_test, num_rounds=num_rounds, local_epochs=local_epochs, sim_name="IID"
    )
    
    # 4. Federated Non-IID
    non_iid_shards = partition_engines_non_iid(train_scaled, num_clients=num_clients, strategy="lifespan")
    local_non_iid_summary = train_local_only(non_iid_shards, feature_cols, X_test, y_test, epochs=num_rounds * local_epochs)
    fed_non_iid_metrics, non_iid_curve = run_federated_simulation(
        non_iid_shards, feature_cols, X_test, y_test, num_rounds=num_rounds, local_epochs=local_epochs, sim_name="Non-IID"
    )
    
    records = [
        {"Setup": "Centralized CNN-LSTM (Upper Bound)", "MAE": cent_metrics["MAE"], "RMSE": cent_metrics["RMSE"], "R2": cent_metrics["R2"], "NASA_Score": cent_metrics["NASA_Score"]},
        {"Setup": "Local-Only Average (IID Shards)", "MAE": local_iid_summary["MAE"], "RMSE": local_iid_summary["RMSE"], "R2": local_iid_summary["R2"], "NASA_Score": local_iid_summary["NASA_Score"]},
        {"Setup": "Local-Only Average (Non-IID Shards)", "MAE": local_non_iid_summary["MAE"], "RMSE": local_non_iid_summary["RMSE"], "R2": local_non_iid_summary["R2"], "NASA_Score": local_non_iid_summary["NASA_Score"]},
        {"Setup": "Federated IID (FedAvg)", "MAE": fed_iid_metrics.get("mae", 0.0), "RMSE": fed_iid_metrics.get("rmse", 0.0), "R2": fed_iid_metrics.get("r2", 0.0), "NASA_Score": fed_iid_metrics.get("nasa_score", 0.0)},
        {"Setup": "Federated Non-IID (FedAvg)", "MAE": fed_non_iid_metrics.get("mae", 0.0), "RMSE": fed_non_iid_metrics.get("rmse", 0.0), "R2": fed_non_iid_metrics.get("r2", 0.0), "NASA_Score": fed_non_iid_metrics.get("nasa_score", 0.0)},
    ]
    
    df_results = pd.DataFrame(records)
    table_path = TABLES_DIR / "federated_comparison.csv"
    df_results.to_csv(table_path, index=False)
    print(f"\nSaved federated comparison table to {table_path}")
    print(df_results.to_string(index=False))
    
    # Save curves
    curves = {"iid": iid_curve, "non_iid": non_iid_curve}
    with open(TABLES_DIR / "fl_convergence_history.json", "w") as f:
        json.dump(curves, f, indent=2)

    # 1. Figure: Federated Convergence Curves (IID vs Non-IID)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if iid_curve:
        rounds_iid = [c["round"] for c in iid_curve]
        mae_iid = [c["mae"] for c in iid_curve]
        rmse_iid = [c["rmse"] for c in iid_curve]
        axes[0].plot(rounds_iid, mae_iid, "o-", label="FedAvg (IID)", color="#1f77b4", linewidth=2)
        axes[1].plot(rounds_iid, rmse_iid, "o-", label="FedAvg (IID)", color="#1f77b4", linewidth=2)
    if non_iid_curve:
        rounds_non = [c["round"] for c in non_iid_curve]
        mae_non = [c["mae"] for c in non_iid_curve]
        rmse_non = [c["rmse"] for c in non_iid_curve]
        axes[0].plot(rounds_non, mae_non, "s--", label="FedAvg (Non-IID)", color="#d62728", linewidth=2)
        axes[1].plot(rounds_non, rmse_non, "s--", label="FedAvg (Non-IID)", color="#d62728", linewidth=2)

    axes[0].axhline(cent_metrics["MAE"], color="green", linestyle=":", label=f"Centralized ({cent_metrics['MAE']:.1f})")
    axes[1].axhline(cent_metrics["RMSE"], color="green", linestyle=":", label=f"Centralized ({cent_metrics['RMSE']:.1f})")

    axes[0].set_title("Global Model Test MAE vs. Communication Round", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Communication Round", fontsize=11)
    axes[0].set_ylabel("MAE (Cycles)", fontsize=11)
    axes[0].legend(frameon=True)

    axes[1].set_title("Global Model Test RMSE vs. Communication Round", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Communication Round", fontsize=11)
    axes[1].set_ylabel("RMSE (Cycles)", fontsize=11)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    conv_fig_path = FIGURES_DIR / "fl_convergence_curves.png"
    plt.savefig(conv_fig_path, dpi=300)
    plt.close()
    print(f"Saved FL convergence curves to {conv_fig_path}")

    # 2. Figure: Centralized vs Local vs Federated Tradeoff Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_results, x="Setup", y="RMSE", palette="mako", ax=axes[0])
    axes[0].set_title("RMSE: Centralized vs. Local vs. Federated", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("RMSE (Cycles)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=25)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    sns.barplot(data=df_results, x="Setup", y="MAE", palette="mako", ax=axes[1])
    axes[1].set_title("MAE: Centralized vs. Local vs. Federated", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE (Cycles)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=25)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    tradeoff_fig_path = FIGURES_DIR / "federated_vs_centralized_tradeoff.png"
    plt.savefig(tradeoff_fig_path, dpi=300)
    plt.close()
    print(f"Saved tradeoff bar chart to {tradeoff_fig_path}")

    return df_results, curves


if __name__ == "__main__":
    run_full_federated_study()
