# utils/setup.py

import os
import torch
from torch import nn
from utils.train_pipeline import EarlyStopper, ModelTrainingPipeline

# Seed for reproducibility
seed = 1729

# Ensure deterministic behavior for reproducibility
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping setup for synthetic data
#early_stopper = EarlyStopper(
#    patience=10, 
#    min_delta=1e-5, 
#    min_epochs=100, 
#    max_time_minutes=15
#)

# Early stopping setup for real data
early_stopper = EarlyStopper(
    patience=10, 
    min_delta=1e-5, 
    min_epochs=1, 
    max_time_minutes=15
)


# Model training pipeline setup
pipeline = ModelTrainingPipeline(device=device, early_stopper=early_stopper)
pipeline.set_seed(seed)

# Example function for debugging
def test_tensor():
    x = torch.rand(5, 3)
    print(x)

# Exporting shared objects
__all__ = ["seed", "device", "early_stopper", "pipeline", "test_tensor"]
