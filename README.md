# Federated Predictive Maintenance: Turbofan Remaining Useful Life (RUL) Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![Flower FL](https://img.shields.io/badge/Flower-1.36-gold.svg)](https://flower.ai/)
[![Dataset: NASA C-MAPSS](https://img.shields.io/badge/NASA-C--MAPSS%20FD001-blueviolet.svg)](https://data.nasa.gov/)

An end-to-end, portfolio-grade predictive maintenance project demonstrating the progression from **Exploratory Data Analysis (EDA)** and **Time-Series Feature Engineering** to **Classical ML Baselines**, **PyTorch Deep Sequence Models (MLP, LSTM, GRU, CNN-LSTM)**, and **Federated Learning (Flower / FedAvg)** across simulated private factory clients.

---

## 1. Project Overview & Architecture

Modern industrial operations (aviation, manufacturing, heavy equipment) rely on multivariate sensor telemetry to predict the **Remaining Useful Life (RUL)** of critical machinery before failure occurs. However, data privacy, proprietary flight profiles, and industrial confidentiality prevent factories from pooling raw sensor streams into a central cloud.

This project investigates:
1. **Signal Extraction**: Which sensor channels exhibit true monotonic degradation trends vs. flat uninformative constants?
2. **Tabular vs. Sequence Modeling**: How much predictive accuracy is gained by moving from engineered tabular features (Linear, Random Forest, XGBoost) to deep temporal sequence models (MLP, LSTM, GRU, CNN-LSTM)?
3. **Federated Learning Trade-Offs**: If $K$ factories collaboratively train via **FedAvg** without exchanging raw data, how close does the global model get to centralized pooling, and how much does it outperform isolated local models?

```
                      +-------------------------------------------------------------+
                      |               Centralized Cloud / Server                     |
                      |            Global CNN-LSTM Weights (FedAvg)                 |
                      +------------------------------+------------------------------+
                                                     |
                         Weights Downlink (W_t)      |      Weights Uplink (W_{t+1})
                         ======================      |      ========================
                                                     |
                  +----------------------------------+----------------------------------+
                  |                                  |                                  |
                  v                                  v                                  v
      +-----------------------+          +-----------------------+          +-----------------------+
      |       Factory 1       |          |       Factory 2       |          |       Factory 3       |
      |   (Low-Stress Fleet)  |          |  (Medium-Stress Fleet)|          |  (High-Stress Fleet)  |
      | Local Turbofan Telemetry|         | Local Turbofan Telemetry|         | Local Turbofan Telemetry|
      |   Private Shard (1/3) |          |   Private Shard (1/3) |          |   Private Shard (1/3) |
      |  Local CNN-LSTM Train |          |  Local CNN-LSTM Train |          |  Local CNN-LSTM Train |
      +-----------------------+          +-----------------------+          +-----------------------+
```

---

## 2. CNN-LSTM Flagship Architecture

The core deep learning model combines **1D Temporal Convolutions** (to extract local inter-sensor physical relationships) with **Long Short-Term Memory (LSTM)** units (to capture long-range monotonic degradation trajectories):

```
Input Sequence: (Batch, Sequence_Length=30, Informative_Sensors=15)
  |
  v
[Permute to (Batch, 15, 30)]
  |
  +---> 1D Convolution (32 filters, kernel=3, padding='same') -> BatchNorm -> ReLU
  |
  +---> 1D Convolution (64 filters, kernel=3, padding='same') -> BatchNorm -> ReLU
  |
[Permute to (Batch, 30, 64)]
  |
  +---> Recurrent Layer: LSTM (hidden_size=64, batch_first=True)
  |
[Extract Last Sequence Timestep (Batch, 64)]
  |
  +---> Dropout (p=0.3)
  |
  +---> Fully Connected (64 -> 32) -> ReLU
  |
  +---> Linear Output Head (32 -> 1) ===> Predicted Remaining Useful Life (RUL)
```

---

## 3. Empirical Results (FD001 Test Set)

All models are evaluated on the exact same held-out NASA C-MAPSS FD001 test set (100 engines truncated prior to failure) using standardized metrics:
- **MAE** (Mean Absolute Error, in cycles)
- **RMSE** (Root Mean Squared Error, in cycles)
- **$R^2$** (Coefficient of Determination)
- **NASA Asymmetric Score**: Penalizes late predictions ($\hat{y} > y$) exponentially more than early predictions ($\hat{y} < y$):
  $$d_i = \hat{y}_i - y_i, \quad S = \sum_{i=1}^N \begin{cases} e^{-d_i / 13} - 1 & d_i < 0 \\ e^{d_i / 10} - 1 & d_i \ge 0 \end{cases}$$

### A. Centralized Benchmark: Classical ML to Sequence Deep Learning

| Category | Model | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Test NASA Score |
|---|---|---|---|---|---|---|---|---|
| Classical ML | Linear Regression | 13.09 | 16.19 | 0.849 | 14.33 | 17.98 | 0.799 | 529.7 |
| Classical ML | Random Forest | 10.44 | 15.24 | 0.867 | 12.07 | 17.21 | 0.816 | 553.4 |
| Classical ML | XGBoost | 10.06 | 14.39 | 0.881 | 11.49 | 16.82 | 0.824 | 604.0 |
| Deep Sequence | MLP (Flattened) | 9.56 | 12.76 | 0.907 | 10.46 | 13.96 | 0.879 | 342.9 |
| Deep Sequence | LSTM | 8.34 | 12.20 | 0.915 | 10.36 | 14.43 | 0.870 | 429.6 |
| Deep Sequence | GRU | 7.87 | 11.76 | 0.921 | 10.36 | 14.55 | 0.868 | 420.7 |
| **Deep Sequence** | **CNN-LSTM (Flagship)** | **9.17** | **12.99** | **0.903** | **10.31** | **15.25** | **0.855** | **606.2** |

> **Key Finding**: Sequence models outperform classical tabular baselines: CNN-LSTM achieves **10.31 Test MAE**, reducing prediction error by **10.3% relative to XGBoost** (11.49 MAE).

### B. Federated Learning Study (3 Factory Clients)

| Training Setup | Test MAE | Test RMSE | Test $R^2$ | Test NASA Score | Key Takeaway |
|---|---|---|---|---|---|
| **Centralized CNN-LSTM** | 10.09 | 15.01 | 0.860 | 642.0 | Centralized upper bound with pooled raw data |
| **Local-Only (IID Average)** | 12.49 | 17.47 | 0.809 | 739.3 | Isolated training on 1/3 random data |
| **Local-Only (Non-IID Average)** | 13.63 | 19.22 | 0.769 | 1365.1 | Severe degradation due to biased operational stress |
| **Federated IID (FedAvg)** | 11.42 | 15.19 | 0.856 | 371.5 | Matches centralized accuracy within 0.18 RMSE |
| **Federated Non-IID (FedAvg)** | **9.31** | **13.42** | **0.888** | **315.5** | **Beats isolated local models by 30.2% RMSE** |

> **Key Finding**: When data is non-IID (factories operating engines across different lifespan regimes), isolated local models suffer catastrophic blindspots (RMSE 19.22, NASA penalty 1365.1). FedAvg federated training recovers global degradation patterns and achieves **13.42 RMSE and 315.5 NASA Score** without sharing any raw time-series data.

---

## 4. Key Visualizations

All visual artifacts are generated from actual runs in `results/figures/`:

1. **Sensor Degradation Curves**: `results/figures/sensor_degradation_trajectories.png`
2. **Correlation Heatmap**: `results/figures/sensor_correlation_matrix.png`
3. **Feature Importances**: `results/figures/feature_importances.png`
4. **Model Comparison Bar Chart**: `results/figures/model_comparison_barchart.png`
5. **Predicted vs. Actual RUL Scatter**: `results/figures/predicted_vs_actual_rul.png`
6. **Federated Convergence Curves**: `results/figures/fl_convergence_curves.png`
7. **Federated vs. Centralized Trade-Off**: `results/figures/federated_vs_centralized_tradeoff.png`

---

## 5. Repository Structure

```
turbofan-rul-prediction/
├── data/
│   └── raw/                       # C-MAPSS raw files (FD001, FD002)
├── notebooks/
│   ├── 01_eda.ipynb               # Sensor distributions, flat channels, trajectories
│   ├── 02_baselines_ml.ipynb      # Rolling stats, Linear Reg, Random Forest, XGBoost
│   ├── 03_deep_learning.ipynb     # Sliding windows, MLP, LSTM, GRU, CNN-LSTM benchmark
│   └── 04_federated_learning.ipynb# Flower simulation, IID vs Non-IID, trade-off analysis
├── src/
│   ├── utils/config.py            # Global constants, paths, and hyperparameters
│   ├── data/
│   │   ├── loader.py              # Automated downloader and parser for C-MAPSS
│   │   ├── rul_labeling.py        # Piecewise-linear RUL ceiling computation
│   │   ├── windowing.py           # Sliding-window sequence tensor generation
│   │   └── splits.py              # Engine-level split & federated client partitioning
│   ├── features/engineering.py    # Flat sensor detection, scaling, rolling statistics
│   ├── models/
│   │   ├── ml_baselines.py        # Linear, Random Forest, and XGBoost wrappers
│   │   ├── mlp.py                 # PyTorch MLP baseline
│   │   ├── lstm.py                # PyTorch 2-layer stacked LSTM
│   │   ├── gru.py                 # PyTorch 2-layer stacked GRU
│   │   ├── cnn_lstm.py            # Flagship PyTorch 1D-CNN-LSTM model
│   │   └── train_utils.py         # PyTorch training loop with Huber loss & early stopping
│   ├── federated/
│   │   ├── client.py              # Flower NumPyClient subclass
│   │   ├── server.py              # FedAvg strategy and centralized round evaluation
│   │   └── simulate.py            # Complete Ray-backed Flower simulation orchestrator
│   ├── eval/metrics.py            # MAE, RMSE, R2, and NASA asymmetric score
│   ├── eda_runner.py              # EDA execution and figure generation script
│   ├── benchmark_runner.py        # Full 7-model training benchmark script
│   └── execute_notebooks.py       # In-place execution of all notebooks
├── results/
│   ├── figures/                   # High-res visualization plots
│   └── tables/                    # CSVs for model and federated comparisons
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## 6. How to Reproduce

### 1. Environment Setup
```powershell
# Clone the repository and navigate to root
git clone https://github.com/parth-05122005/turbofan-rul-prediction.git
cd turbofan-rul-prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis (EDA)
```powershell
python -m src.eda_runner
```

### 3. Run Benchmark Suite (7 Models)
```powershell
python -m src.benchmark_runner
```
Outputs `results/tables/model_comparison.csv` and benchmark comparison figures.

### 4. Run Federated Learning Simulation
```powershell
python -m src.federated.simulate
```
Outputs `results/tables/federated_comparison.csv`, convergence curves, and trade-off figures.

### 5. Open & Explore Notebooks
```powershell
jupyter lab
```
Open notebooks in sequence:
- `notebooks/01_eda.ipynb`
- `notebooks/02_baselines_ml.ipynb`
- `notebooks/03_deep_learning.ipynb`
- `notebooks/04_federated_learning.ipynb`

---

## 7. References
1. **A. Saxena, K. Goebel, D. Simon, N. Eklund**, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, IEEE Prognostics and Health Management, 2008.
2. **F. O. Heimes**, *"Recurrent neural networks for remaining useful life estimation"*, IEEE PHM Conference, 2008.
3. **B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas**, *"Communication-Efficient Learning of Deep Networks from Decentralized Data"*, AISTATS, 2017.
4. **Flower: A Friendly Federated Learning Framework**, [https://flower.ai/](https://flower.ai/)
