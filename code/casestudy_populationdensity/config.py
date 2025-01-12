# config.py

from pathlib import Path

# Base paths and configuration
BASE_DIR = Path('../../').resolve()
CASE_STUDY = 'populationdensity'
DATASET = 'hourly_township'
TOWNSHIP_NAME = "Avenidas Novas"
VARIABLES = ["sum_terminals", "sum_roaming_terminals", "sum_phonecalls"]

STUMPY_EXCL_ZONE_DENOM = 2  # r = np.ceil(m/2)
TOP_K_MP = 1
INCLUDE = None
NORMALIZE = True
SUBSQUENCES_LENGTHS = [4,6,12]
NORMALIZE_FLAGS = {"X_series": True, "X_mask": False, "X_indices": True}


# Derived directories
RESULTS_DIR = BASE_DIR / 'results' / CASE_STUDY 
RESULTS_MOTIF_DIR = RESULTS_DIR / 'mp'/ DATASET 
IMAGES_DIR = BASE_DIR / 'images' / CASE_STUDY  
DATA_DIR = BASE_DIR / 'data' / CASE_STUDY


DATASET_PATH = DATA_DIR / f"{DATASET}.csv"

# Ensure directories exist
RESULTS_MOTIF_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Results will be saved in: {RESULTS_DIR}")
logging.info(f"Images will be saved in: {IMAGES_DIR}")
logging.info(f"Data will be accessed from: {DATA_DIR}")


