
SEED = 42

VITAL_COLUMNS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "respiratory_rate",
    "oxygen_saturation",
]

# Experiments
WINDOW_SIZES = [2, 12, 24]
PREDICTION_HORIZONS = [1, 2, 4]

# Training
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-4

# Model dimensions
INPUT_DIM = len(VITAL_COLUMNS)
HIDDEN_DIM = 64
OUTPUT_DIM = len(VITAL_COLUMNS)
DROPOUT = 0.1

# ARIMA
ARIMA_ORDER = (1, 0, 0)
ARIMA_MIN_SERIES_LENGTH = 10

# Paths
TRAIN_DATA_PATH = "data/processed/splits/train.csv"
VAL_DATA_PATH = "data/processed/splits/val.csv"
TEST_DATA_PATH = "data/processed/splits/test.csv"

METRICS_DIR = "outputs/metrics"
PLOTS_DIR = "outputs/plots"
TABLES_DIR = "outputs/tables"

ARIMA_OUTPUT_PATH = "outputs/metrics/arima_results.csv"
