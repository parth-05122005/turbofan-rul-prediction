"""
Configuration constants and paths for Federated Predictive Maintenance.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure essential directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset Column Specifications (NASA C-MAPSS)
ID_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# In FD001, these sensors have zero or near-zero variance across all engines and operational settings
FLAT_SENSORS_FD001 = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

INFORMATIVE_SENSORS_FD001 = [s for s in SENSOR_COLUMNS if s not in FLAT_SENSORS_FD001]

# Default Hyperparameters
RUL_CAP = 125          # Piecewise linear RUL threshold (Heimes 2008 standard)
SEQUENCE_LENGTH = 30   # Sliding window length W
RANDOM_SEED = 42

# Remote dataset mirror
CMAPSS_BASE_URL = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"
