# config.py

from pathlib import Path

# Base paths and configuration
BASE_DIR = Path('../../').resolve()
CASE_STUDY = 'syntheticdata2'
VARIABLES = '[0,2]'

# Derived directories
RESULTS_DIR = BASE_DIR / 'results' / CASE_STUDY / f"variables={VARIABLES}"
IMAGES_DIR = BASE_DIR / 'images' / CASE_STUDY / f"variables={VARIABLES}"
DATA_DIR = BASE_DIR / 'data' / CASE_STUDY / f"variables={VARIABLES}"

# Dataset file paths
N = 100000  # Number of data points
K = 3       # Number of variables
P = 5       # Pattern length
VARIABLES_PATTERN = [0,2]
NORMALIZE_FLAGS = {"X_series": True, "X_mask": False, "X_indices": True}

DATASET_PATH = DATA_DIR / f"scenario2_n={N}_k={K}_p={P}_max_interval=45_variables={VARIABLES_PATTERN}.csv"
MOTIF_INDEXES_PATH = DATA_DIR / f"motif_indexes_scenario2_n={N}_k={K}_p={P}_max_interval=45.csv"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Results will be saved in: {RESULTS_DIR}")
logging.info(f"Images will be saved in: {IMAGES_DIR}")
logging.info(f"Data will be accessed from: {DATA_DIR}")
