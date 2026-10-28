"""
Comprehensive benchmark runner training and evaluating:
- 3 Classical ML Baselines: Linear Regression, Random Forest, XGBoost
- 4 Deep Sequence Models: MLP, LSTM, GRU, CNN-LSTM
Saves master table to results/tables/model_comparison.csv and generates comparison figures.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.utils.config import (
    RUL_CAP,
    SEQUENCE_LENGTH,
    RANDOM_SEED,
    TABLES_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
)
from src.data.loader import load_dataset
from src.data.rul_labeling import add_rul_labels, add_test_rul_labels
from src.data.splits import train_val_split_by_engine
from src.features.engineering import (
    get_informative_features,
    fit_transform_scaler,
    extract_tabular_features,
)
from src.data.windowing import (
    generate_sliding_windows,
    generate_test_windows,
    TurbofanDataset,
)
from src.models.ml_baselines import get_ml_models
from src.models.mlp import MLPRegressor
from src.models.lstm import LSTMRegressor
from src.models.gru import GRURegressor
from src.models.cnn_lstm import CNNLSTMRegressor
from src.models.train_utils import train_deep_model, evaluate_model_on_windows
from src.eval.metrics import compute_all_metrics

# Set random seeds for strict reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def run_benchmark():
    print("=" * 70)
    print("STARTING END-TO-END PREDICTIVE MAINTENANCE BENCHMARK (FD001)")
    print("=" * 70)

    # 1. Load Data
    train_df, test_df, rul_df = load_dataset("FD001")
    feature_cols = get_informative_features(train_df, include_settings=False)
    print(f"Informative Sensor Count: {len(feature_cols)} -> {feature_cols}")

    train_df_rul = add_rul_labels(train_df, rul_cap=RUL_CAP, piecewise=True)
    test_df_rul = add_test_rul_labels(test_df, rul_df, rul_cap=RUL_CAP, piecewise=True)

    # Scaled data
    train_scaled, test_scaled, scaler = fit_transform_scaler(
        train_df_rul, test_df_rul, feature_cols, scaler_type="minmax"
    )

    # Train / Validation engine split
    train_split, val_split = train_val_split_by_engine(train_scaled, val_ratio=0.2, random_state=RANDOM_SEED)
    print(f"Train engines: {train_split['unit_number'].nunique()}, Val engines: {val_split['unit_number'].nunique()}")

    # =========================================================================
    # PART 1: CLASSICAL ML BASELINES (Tabular Features)
    # =========================================================================
    print("\n--- Feature Engineering for Classical ML Baselines ---")
    tr_eng = extract_tabular_features(train_split, feature_cols)
    va_eng = extract_tabular_features(val_split, feature_cols)
    te_eng = extract_tabular_features(test_scaled, feature_cols)

    # For test set evaluation: use the last row of each test engine
    te_last = te_eng.groupby("unit_number").last().reset_index()

    tab_cols = [c for c in tr_eng.columns if c not in ["unit_number", "time_in_cycles", "setting_1", "setting_2", "setting_3", "raw_rul", "RUL"]]
    print(f"Engineered Tabular Features: {len(tab_cols)}")

    X_tr_tab = tr_eng[tab_cols].values
    y_tr_tab = tr_eng["RUL"].values
    X_va_tab = va_eng[tab_cols].values
    y_va_tab = va_eng["RUL"].values
    X_te_tab = te_last[tab_cols].values
    y_te_tab = te_last["RUL"].values

    ml_models = get_ml_models()
    all_results = []
    trained_models = {}
    test_predictions = {}

    for name, model in ml_models.items():
        print(f"Fitting {name}...")
        model.fit(X_tr_tab, y_tr_tab)
        trained_models[name] = model

        va_preds = model.predict(X_va_tab)
        te_preds = model.predict(X_te_tab)
        test_predictions[name] = te_preds

        va_m = compute_all_metrics(y_va_tab, va_preds)
        te_m = compute_all_metrics(y_te_tab, te_preds)

        print(f"  -> Test MAE: {te_m['MAE']:.2f}, RMSE: {te_m['RMSE']:.2f}, R2: {te_m['R2']:.3f}, NASA: {te_m['NASA_Score']:.1f}")
        all_results.append({
            "Category": "Classical ML",
            "Model": name,
            "Val_MAE": va_m["MAE"],
            "Val_RMSE": va_m["RMSE"],
            "Val_R2": va_m["R2"],
            "Test_MAE": te_m["MAE"],
            "Test_RMSE": te_m["RMSE"],
            "Test_R2": te_m["R2"],
            "Test_NASA_Score": te_m["NASA_Score"],
        })

    # Save feature importances from Random Forest & XGBoost
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (m_name, m_obj) in enumerate([("Random Forest", trained_models["Random Forest"]), ("XGBoost", trained_models["XGBoost"])]):
        imp = pd.Series(m_obj.feature_importances_, index=tab_cols).sort_values(ascending=False).head(15)
        sns.barplot(x=imp.values, y=imp.index, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Top 15 Features ({m_name})", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Relative Importance", fontsize=11)

    plt.tight_layout()
    feat_imp_path = FIGURES_DIR / "feature_importances.png"
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"Saved feature importances plot to {feat_imp_path}")

    # =========================================================================
    # PART 2: DEEP LEARNING SEQUENCE MODELS (Sliding Windows)
    # =========================================================================
    print("\n--- Sliding Window Generation for Deep Sequence Models ---")
    X_tr_seq, y_tr_seq = generate_sliding_windows(train_split, feature_cols, "RUL", SEQUENCE_LENGTH)
    X_va_seq, y_va_seq = generate_sliding_windows(val_split, feature_cols, "RUL", SEQUENCE_LENGTH)
    X_te_seq, y_te_seq, te_units = generate_test_windows(
        test_scaled, rul_df, feature_cols, seq_length=SEQUENCE_LENGTH, rul_cap=RUL_CAP, piecewise=True
    )
    print(f"Sequence Shapes -> Train: {X_tr_seq.shape}, Val: {X_va_seq.shape}, Test: {X_te_seq.shape}")

    train_loader = DataLoader(TurbofanDataset(X_tr_seq, y_tr_seq), batch_size=64, shuffle=True)
    val_loader = DataLoader(TurbofanDataset(X_va_seq, y_va_seq), batch_size=64, shuffle=False)

    deep_models = {
        "MLP (Flattened)": MLPRegressor(seq_len=SEQUENCE_LENGTH, n_features=len(feature_cols)),
        "LSTM": LSTMRegressor(n_features=len(feature_cols), hidden_dim=64, num_layers=2),
        "GRU": GRURegressor(n_features=len(feature_cols), hidden_dim=64, num_layers=2),
        "CNN-LSTM (Flagship)": CNNLSTMRegressor(n_features=len(feature_cols)),
    }

    deep_histories = {}
    best_cnn_lstm = None

    for d_name, d_model in deep_models.items():
        print(f"\nTraining Deep Model: {d_name}...")
        fitted_model, hist = train_deep_model(
            d_model,
            train_loader,
            val_loader,
            epochs=40,
            lr=1e-3,
            patience=8,
            device="cpu",
            verbose=False,
        )
        deep_histories[d_name] = hist
        
        # Evaluate on test set
        te_preds, te_metrics = evaluate_model_on_windows(fitted_model, X_te_seq, y_te_seq, device="cpu")
        test_predictions[d_name] = te_preds
        print(f"  -> Test MAE: {te_metrics['MAE']:.2f}, RMSE: {te_metrics['RMSE']:.2f}, R2: {te_metrics['R2']:.3f}, NASA: {te_metrics['NASA_Score']:.1f}")

        # Compute validation metrics on best weights
        _, va_metrics = evaluate_model_on_windows(fitted_model, X_va_seq, y_va_seq, device="cpu")

        all_results.append({
            "Category": "Deep Sequence",
            "Model": d_name,
            "Val_MAE": va_metrics["MAE"],
            "Val_RMSE": va_metrics["RMSE"],
            "Val_R2": va_metrics["R2"],
            "Test_MAE": te_metrics["MAE"],
            "Test_RMSE": te_metrics["RMSE"],
            "Test_R2": te_metrics["R2"],
            "Test_NASA_Score": te_metrics["NASA_Score"],
        })

        if d_name == "CNN-LSTM (Flagship)":
            best_cnn_lstm = fitted_model
            torch.save(fitted_model.state_dict(), RESULTS_DIR / "cnn_lstm_best.pt")
            print(f"Saved CNN-LSTM checkpoint to {RESULTS_DIR / 'cnn_lstm_best.pt'}")

    # =========================================================================
    # PART 3: MASTER COMPARISON TABLE & FIGURES
    # =========================================================================
    df_comparison = pd.DataFrame(all_results)
    table_file = TABLES_DIR / "model_comparison.csv"
    df_comparison.to_csv(table_file, index=False)
    print("\n" + "=" * 70)
    print("MASTER MODEL COMPARISON TABLE (FD001 TEST SET)")
    print("=" * 70)
    print(df_comparison.to_string(index=False))

    # Figure 1: Model Comparison Bar Chart (MAE & RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = ["#4C72B0" if cat == "Classical ML" else "#55A868" for cat in df_comparison["Category"]]
    
    sns.barplot(data=df_comparison, x="Model", y="Test_RMSE", palette=palette, ax=axes[0])
    axes[0].set_title("Test RMSE Across Models (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("RMSE (Cycles)", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    sns.barplot(data=df_comparison, x="Model", y="Test_MAE", palette=palette, ax=axes[1])
    axes[1].set_title("Test MAE Across Models (Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MAE (Cycles)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    plt.suptitle("Model Benchmark Progression: Classical ML to Deep Sequence Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    barchart_path = FIGURES_DIR / "model_comparison_barchart.png"
    plt.savefig(barchart_path, dpi=300)
    plt.close()
    print(f"Saved comparison bar chart to {barchart_path}")

    # Figure 2: Deep Learning Training / Validation Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (m_name, hist) in enumerate(deep_histories.items()):
        ax = axes[idx]
        ax.plot(hist["train_loss"], label="Train Huber Loss", color="#1f77b4", linewidth=2)
        ax.plot(hist["val_loss"], label="Val Huber Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        ax.set_title(m_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.legend(frameon=True)
    plt.suptitle("Training and Validation Convergence of Deep Sequence Models", fontsize=15, fontweight="bold")
    plt.tight_layout()
    loss_curve_path = FIGURES_DIR / "deep_learning_loss_curves.png"
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Saved loss curves to {loss_curve_path}")

    # Figure 3: Predicted vs Actual RUL Scatter Plot for CNN-LSTM
    best_preds = test_predictions["CNN-LSTM (Flagship)"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_te_seq, best_preds, color="#2b5c8f", alpha=0.7, edgecolors="k", s=50, label="Test Engines (N=100)")
    ax.plot([0, RUL_CAP + 10], [0, RUL_CAP + 10], "r--", linewidth=2, label="Perfect Calibration Line")

    # Add callouts for sample engines
    callout_indices = [5, 27, 49, 75, 91]
    for idx in callout_indices:
        if idx < len(y_te_seq):
            u_id = te_units[idx]
            act = y_te_seq[idx]
            prd = best_preds[idx]
            ax.annotate(
                f"Unit {u_id}\nAct: {act:.0f}, Pred: {prd:.0f}",
                (act, prd),
                xytext=(act + 5, prd - 10),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.4),
            )

    ax.set_title("Predicted vs. Actual RUL on Test Set (CNN-LSTM Flagship)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual Ground Truth RUL (Cycles)", fontsize=12)
    ax.set_ylabel("Predicted RUL (Cycles)", fontsize=12)
    ax.legend(loc="upper left", frameon=True)
    pred_path = FIGURES_DIR / "predicted_vs_actual_rul.png"
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"Saved predicted vs actual plot to {pred_path}")

    print("\nBENCHMARK RUN COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_benchmark()
