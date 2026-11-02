"""Project-wide constants: paths, split boundaries and random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "monthly_consumption.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Split boundaries refer to the TARGET month (the month being forecast).
# Features for a target month only use readings from strictly earlier months.
TRAIN_END = "2024-04-30"   # train targets: 2023-04 .. 2024-04 (13 months)
VAL_END = "2024-07-31"     # validation targets: 2024-05 .. 2024-07 (3 months)
# test targets: 2024-08 .. 2024-10 (3 months)

# Last month whose readings may be used to fit cleaning statistics
# (household medians, global median, outlier cap). Equal to TRAIN_END so the
# validation/test periods never influence preprocessing.
CLEANING_FIT_END = TRAIN_END

MIN_VALID_MONTHS = 12      # households with fewer valid readings are dropped
CAP_QUANTILE = 0.995       # quantile for the 'global' outlier cap option
SPIKE_MULTIPLE = 5.0       # 'household' option: clip readings > 5x the household median
MAPE_MIN_ACTUAL = 10.0     # kWh; actuals below this are excluded from MAPE

LAGS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6]
MIN_TARGET_INDEX = 6       # first target month with full lag_6 / 6-month window
