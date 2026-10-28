"""
Script to execute complete Exploratory Data Analysis (EDA) and save figures.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.utils.config import (
    FIGURES_DIR,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    FLAT_SENSORS_FD001,
    INFORMATIVE_SENSORS_FD001,
    RUL_CAP,
)
from src.data.loader import load_dataset
from src.data.rul_labeling import add_rul_labels
from src.features.engineering import identify_flat_sensors

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def run_eda():
    print("Loading FD001 dataset for EDA...")
    train_df, test_df, rul_df = load_dataset("FD001")
    train_df_rul = add_rul_labels(train_df, rul_cap=RUL_CAP, piecewise=True)

    # 1. Check flat sensors empirically
    flat_detected = identify_flat_sensors(train_df, std_threshold=1e-4)
    print(f"Empirically detected flat sensors (std < 1e-4): {flat_detected}")

    # 2. Cycle-length distribution (Initial engine lifespans)
    max_cycles = train_df.groupby("unit_number")["time_in_cycles"].max()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(max_cycles, bins=25, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Engine Lifespan Distribution (Run-to-Failure Cycles in FD001)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Max Cycles to Failure", fontsize=12)
    ax.set_ylabel("Engine Count", fontsize=12)
    ax.axvline(max_cycles.mean(), color="crimson", linestyle="--", label=f"Mean: {max_cycles.mean():.1f}")
    ax.axvline(max_cycles.median(), color="darkorange", linestyle=":", label=f"Median: {max_cycles.median():.1f}")
    ax.legend(frameon=True)
    cycle_dist_path = FIGURES_DIR / "cycle_length_distribution.png"
    plt.savefig(cycle_dist_path, dpi=300)
    plt.close()
    print(f"Saved {cycle_dist_path}")

    # 3. Correlation Heatmap across 21 sensors + 3 settings
    fig, ax = plt.subplots(figsize=(16, 12))
    corr_cols = SETTING_COLUMNS + SENSOR_COLUMNS
    corr_mat = train_df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap: Operating Settings and Sensors (FD001)", fontsize=16, fontweight="bold")
    corr_path = FIGURES_DIR / "sensor_correlation_matrix.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved {corr_path}")

    # 4. Degradation Trajectories for Sample Engines
    sample_units = [1, 2, 3, 4, 5]
    top_informative = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for idx, s in enumerate(top_informative):
        ax = axes[idx]
        for u in sample_units:
            unit_data = train_df[train_df["unit_number"] == u]
            ax.plot(unit_data["time_in_cycles"], unit_data[s], label=f"Engine {u}", alpha=0.8, linewidth=1.5)
        ax.set_title(f"Degradation Signal: {s}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Raw Reading", fontsize=11)
        if idx >= 3:
            ax.set_xlabel("Time in Cycles", fontsize=11)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle("Run-to-Failure Sensor Degradation Trajectories (Sample Engines)", fontsize=16, fontweight="bold")
    deg_path = FIGURES_DIR / "sensor_degradation_trajectories.png"
    plt.savefig(deg_path, dpi=300)
    plt.close()
    print(f"Saved {deg_path}")

    # 5. Sensor vs RUL Monotonicity
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, s in enumerate(top_informative):
        ax = axes[idx]
        sns.lineplot(
            data=train_df_rul,
            x="RUL",
            y=s,
            ax=ax,
            color="#2ca02c",
            errorbar="sd",
            linewidth=2,
        )
        ax.invert_xaxis()  # RUL decreases towards failure (0)
        ax.set_title(f"{s} vs. Remaining Useful Life", fontsize=13, fontweight="bold")
        ax.set_xlabel("RUL (Cycles remaining -> 0 = Failure)", fontsize=11)
        ax.set_ylabel("Sensor Value", fontsize=11)

    fig.suptitle("Mean Sensor Trajectory vs. Remaining Useful Life (Capped RUL)", fontsize=16, fontweight="bold")
    mono_path = FIGURES_DIR / "sensor_vs_rul_monotonicity.png"
    plt.savefig(mono_path, dpi=300)
    plt.close()
    print(f"Saved {mono_path}")

    # 6. Operating Conditions in FD002 (Demonstrating multi-regime clustering)
    train_fd002, _, _ = load_dataset("FD002")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(train_fd002[SETTING_COLUMNS])
    train_fd002["regime_cluster"] = clusters

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        train_fd002["setting_1"],
        train_fd002["setting_2"],
        c=train_fd002["regime_cluster"],
        cmap="tab10",
        alpha=0.6,
        s=15,
    )
    ax.set_title("Operating Regime Clustering in FD002 (6 Discrete Regimes)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Operational Setting 1 (Altitude / Mach)", fontsize=12)
    ax.set_ylabel("Operational Setting 2 (Throttle Resolver Angle)", fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Operating Regime Cluster ID", fontsize=11)
    regime_path = FIGURES_DIR / "operating_conditions_clustering.png"
    plt.savefig(regime_path, dpi=300)
    plt.close()
    print(f"Saved {regime_path}")

    print("EDA execution and plot generation completed successfully!")


if __name__ == "__main__":
    run_eda()
