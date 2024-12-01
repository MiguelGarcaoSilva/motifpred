# %%
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import optuna
import random

results_dir = '../results'

# %%
import torch
from torch import nn
import torch.optim as optim

seed = 1729

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

set_seed(seed)

x = torch.rand(5, 3)
print(x)

# %%
from torch.nn.utils.rnn import pad_sequence

def create_dataset(data, variable_indexes, lookback_period, step, forecast_period, motif_indexes):
    X1, X2, y = [], [], []  # X1: data, X2: indexes of the motifs, y: distance to the next motif
    
    for idx in range(len(data[0]) - lookback_period - 1):
        if idx % step != 0:
            continue

        window_end_idx = idx + lookback_period
        forecast_period_end = window_end_idx + forecast_period

        # If there are no more matches after the window, break
        if not any([window_end_idx < motif_idx for motif_idx in motif_indexes]):
            break

        # Motif indexes in window, relative to the start of the window
        motif_indexes_in_window = [motif_idx - idx for motif_idx in motif_indexes if idx <= motif_idx <= window_end_idx]
        motif_indexes_in_forecast_period = [motif_idx for motif_idx in motif_indexes if window_end_idx < motif_idx <= forecast_period_end]

        if motif_indexes_in_forecast_period:
            next_match_in_forecast_period = motif_indexes_in_forecast_period[0]
        else:
            next_match_in_forecast_period = -1  # No match in the forecast period but exists in the future

        # Get the data window and transpose to (lookback_period, num_features)
        data_window = data[variable_indexes, idx:window_end_idx].T

        # Calculate `y`
        data_y = -1
        if next_match_in_forecast_period != -1:
            # Index of the next match relative to the end of the window
            data_y = next_match_in_forecast_period - window_end_idx
        
        # Append to lists
        X1.append(torch.tensor(data_window, dtype=torch.float32))  # Now with shape (lookback_period, num_features)
        X2.append(torch.tensor(motif_indexes_in_window, dtype=torch.long)) 
        y.append(data_y) 

    # Pad X2 sequences to have the same length
    X2_padded = pad_sequence(X2, batch_first=True, padding_value=-1) # Final shape: (num_samples, max_num_motifs)
    
    # Convert lists to torch tensors
    X1 = torch.stack(X1)  # Final shape: (num_samples, lookback_period, num_features)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    return X1, X2_padded, y


# %%
#load data 
n = 100000 #number of data points
k = 3 #number of variables
p = 5 # pattern length
variable_indexes = range(k)

data_scenario3 = np.genfromtxt("../data/syntheticdata/scenario3_n={}_k={}_p={}_min_step={}_max_step={}.csv".format(n, k, p, 5, 45), delimiter=",").astype(int).reshape((k, n))

motif_indexes_scenario3 = np.genfromtxt("../data/syntheticdata/motif_indexes_scenario3_n={}_k={}_p={}_min_step={}_max_step={}.csv".format(n, k, p, 5, 45), delimiter=",").astype(int)

print(motif_indexes_scenario3)


# %%
from timeseries_split import BlockingTimeSeriesSplit

#create index  
indexes = np.arange(len(data_scenario3[0]))

#split data
tscv = BlockingTimeSeriesSplit(n_splits=5)
# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))
for i, (train_index, test_index) in enumerate(tscv.split(indexes)):
    # Plot train and test indices
    ax.plot(train_index, np.zeros_like(train_index) + i, 'o', color='lightblue')
    ax.plot(test_index, np.zeros_like(test_index) + i, 'o', color='red')
    print("TRAIN:", train_index, "TEST:", test_index)
    

ax.set_yticks(np.arange(5), ["Fold {}".format(i) for i in range(1, 6)])
plt.show()

# %%
lookback_period = 100 #window size
step = 5 #step size for the sliding window
forecast_period = 50 #forward window size

#x1: past window, x2: indexes of the motif in the window,  y: next relative index of the motif
X1, X2, y = create_dataset(data_scenario3, variable_indexes, lookback_period, step, forecast_period, motif_indexes_scenario3)

# X1, X2, and y are now PyTorch tensors
print("X1 shape:", X1.shape)  # Expected shape: (num_samples, lookback_period, num_features)
print("X2 shape:", X2.shape)  # Expected shape: (num_samples, max_motif_length_in_window)
print("y shape:", y.shape)    # Expected shape: (num_samples, 1)


# %%
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import optuna
import joblib
from typing import Tuple, List
import csv
from models.lstm_pytorch import LSTMX1Input, LSTMX1_X2BeforeLSTM, LSTMX1_X2AfterLSTM

# Helper classes and functions
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.min_epochs = min_epochs

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:  # if the validation loss improves compared to the best 
            self.min_validation_loss = validation_loss
            self.counter = 0  
        elif validation_loss > (self.min_validation_loss + self.min_delta): # if the validation loss is worse than the best + min_delta
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def evaluate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    return mae, rmse

# Scale training data and apply the scaler to validation and test data
def scale_data(X_train, X_val, X_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = torch.tensor(scaler.fit_transform(X_train.view(-1, X_train.shape[-1])), dtype=torch.float32).view(X_train.shape)
    X_val = torch.tensor(scaler.transform(X_val.view(-1, X_val.shape[-1])), dtype=torch.float32).view(X_val.shape)
    X_test = torch.tensor(scaler.transform(X_test.view(-1, X_test.shape[-1])), dtype=torch.float32).view(X_test.shape)
    return X_train, X_val, X_test


def train_model_single_input(model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, early_stopper=None):
    best_val_loss = float('inf')
    best_model_state = None  # Store the state of the best model
    train_losses = []  # To store training loss for every epoch
    validation_losses = []  # To store validation loss for every epoch

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0  # Accumulate training loss for the epoch
        for batch_X1, batch_y in train_loader:
            batch_X1, batch_y = batch_X1.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X1)
            loss = criterion(predictions, batch_y)
            epoch_train_loss += loss.item()  # Accumulate loss
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Append average train loss for this epoch

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X1, batch_y in val_loader:
                batch_X1, batch_y = batch_X1.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_X1), batch_y).item()
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)  # Append average validation loss for this epoch

        # Check early stopping criteria            
        if early_stopper is not None and epoch >= early_stopper.min_epochs and early_stopper.early_stop(avg_val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save best model state

    # Load the best model state before returning, to ensure best validation performance
    model.load_state_dict(best_model_state)

    return best_val_loss, model, train_losses, validation_losses  # Return train losses along with model and validation loss



def train_model_dual_input(model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, early_stopper=None):
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []  # Track training losses per epoch
    validation_losses = []  # Track validation losses per epoch

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X1, batch_X2, batch_y in train_loader:
            batch_X1, batch_X2, batch_y = batch_X1.to(device), batch_X2.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X1, batch_X2)
            loss = criterion(predictions, batch_y)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Append training loss for this epoch

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X1, batch_X2, batch_y in val_loader:
                batch_X1, batch_X2, batch_y = batch_X1.to(device), batch_X2.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_X1, batch_X2), batch_y).item()
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Early stopping
        if early_stopper is not None and epoch >= early_stopper.min_epochs and early_stopper.early_stop(avg_val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    return best_val_loss, model, train_losses, validation_losses  # Return training losses




def run_cross_val(trial, results_folder, model_class, X1, y, X2=None, criterion=torch.nn.MSELoss(), num_epochs=500):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    fold_results, test_mae_per_fold, test_rmse_per_fold, test_losses = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(BlockingTimeSeriesSplit(n_splits=5).split(X1)):
        early_stopper = EarlyStopper(patience=10, min_delta=1e-5, min_epochs=100)

        train_val_split_idx = int(0.8 * len(train_idx))
        train_idx, val_index = train_idx[:train_val_split_idx], train_idx[train_val_split_idx:]
    
        X1_train, X1_val, X1_test, y_train, y_val, y_test = X1[train_idx], X1[val_index], X1[test_idx], y[train_idx], y[val_index], y[test_idx]
        X1_train_scaled, X1_val_scaled, X1_test_scaled = scale_data(X1_train, X1_val, X1_test)

        if X2 is not None:
            X2_train, X2_val, X2_test = X2[train_idx], X2[val_index], X2[test_idx]
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(TensorDataset(X1_train_scaled, X2_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X1_val_scaled, X2_val, y_val), batch_size=len(X1_val_scaled), shuffle=False)
            test_loader = DataLoader(TensorDataset(X1_test_scaled, X2_test, y_test), batch_size=len(X1_test_scaled), shuffle=False)

            model = model_class(input_size=X1.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1, auxiliary_input_dim=X2.shape[1]).to(device)
            fold_val_loss, model, train_losses, validation_losses = train_model_dual_input(model, criterion, torch.optim.Adam(model.parameters(), lr=learning_rate), train_loader, val_loader, num_epochs=num_epochs, early_stopper=early_stopper)

        else:
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(TensorDataset(X1_train_scaled, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X1_val_scaled, y_val), batch_size=len(X1_val_scaled), shuffle=False)
            test_loader = DataLoader(TensorDataset(X1_test_scaled, y_test), batch_size=len(X1_test_scaled), shuffle=False)

            model = model_class(input_size=X1.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
            fold_val_loss, model, train_losses, validation_losses = train_model_single_input(model, criterion, torch.optim.Adam(model.parameters(), lr=learning_rate), train_loader, val_loader, num_epochs=num_epochs, early_stopper=early_stopper)

        fold_results.append(fold_val_loss)
        trial.set_user_attr(f"fold_{fold + 1}_train_losses", train_losses)
        trial.set_user_attr(f"fold_{fold + 1}_validation_losses", validation_losses)

        # Test evaluation
        all_predictions, all_true_values = [], []
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                if X2 is not None:
                    batch_X1, batch_X2, batch_y = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
                    predictions = model(batch_X1, batch_X2)
                else:
                    batch_X1, batch_y = batch_data[0].to(device), batch_data[1].to(device)
                    predictions = model(batch_X1)
                test_loss += criterion(predictions, batch_y).item()
                all_predictions.append(predictions)
                all_true_values.append(batch_y)

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        all_predictions = torch.cat(all_predictions)
        all_true_values = torch.cat(all_true_values)
        mae, rmse = evaluate_metrics(all_predictions, all_true_values)
        test_mae_per_fold.append(mae)
        test_rmse_per_fold.append(rmse)

    mean_val_loss = np.mean(fold_results)
    mean_test_loss = np.mean(test_losses)
    mean_test_mae, std_test_mae = np.mean(test_mae_per_fold), np.std(test_mae_per_fold)
    mean_test_rmse, std_test_rmse = np.mean(test_rmse_per_fold), np.std(test_rmse_per_fold)

    # Log metrics to Optuna
    trial.set_user_attr("fold_val_losses", fold_results)
    trial.set_user_attr("mean_val_loss", mean_val_loss)
    trial.set_user_attr("test_losses", test_losses)
    trial.set_user_attr("mean_test_loss", mean_test_loss)
    trial.set_user_attr("mean_test_mae", mean_test_mae)
    trial.set_user_attr("std_test_mae", std_test_mae)
    trial.set_user_attr("mean_test_rmse", mean_test_rmse)
    trial.set_user_attr("std_test_rmse", std_test_rmse)


    return mean_val_loss, mean_test_loss, model




def run_optuna_study(objective_func, model_class, X1, y, results_folder: str, n_trials: int = 100, num_epochs=500, X2=None):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    file_name = os.path.join(results_folder, "study.pkl")
    
    def objective(trial):
        criterion = torch.nn.MSELoss()  # Define the criterion here
        trial_val_loss, _, _ = objective_func(trial, results_folder, model_class, X1, y, X2, criterion, num_epochs)  # Pass criterion

        return trial_val_loss

    # Let Optuna manage trials and pass them to the objective function
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, file_name)
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(results_folder, "study_results.csv"), index=False)

    print("Best hyperparameters:", study.best_params)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from models.lstm_pytorch import LSTMX1Input

set_seed(seed)

n_trials = 50
num_epochs = 500

result_dir = os.path.join(results_dir, f"LSTMX1Input_{n_trials}_trials_{num_epochs}_epochs")
os.makedirs(result_dir, exist_ok=True)  # Create directory if not exists

# Construct result paths
run_optuna_study(run_cross_val, LSTMX1Input, X1, y, result_dir, n_trials=n_trials, num_epochs=num_epochs)

study = joblib.load(os.path.join(result_dir, "study.pkl"))
study_df = study.trials_dataframe()

print("Validation Losses", study.best_trial.user_attrs["fold_val_losses"])
print("Mean validation loss:", study.best_trial.user_attrs["mean_val_loss"])
print("Test Losses", study.best_trial.user_attrs["test_losses"]) 
print("Mean test loss:", study.best_trial.user_attrs["mean_test_loss"])
print("Mean test MAE:", study.best_trial.user_attrs["mean_test_mae"])
print("Mean test RMSE:", study.best_trial.user_attrs["mean_test_rmse"])

#get the best model results from study_df and for each fold plot the training loss
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

best_fold_train_losses = []
best_fold_val_losses = []

for i in range(5):
    # Extract losses for the best trial for the current fold
    best_fold_train_losses.append(study_df[f"user_attrs_fold_{i + 1}_train_losses"].iloc[study_df["value"].idxmin()])
    best_fold_val_losses.append(study_df[f"user_attrs_fold_{i + 1}_validation_losses"].iloc[study_df["value"].idxmin()])
    
    # Plot train and validation losses in the current subplot
    axes[i].plot(best_fold_train_losses[i], label="Train Loss")
    axes[i].plot(best_fold_val_losses[i], label="Validation Loss")
    
    # Customize the subplot
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:  # Only set ylabel for the first subplot
        axes[i].set_ylabel("Loss")
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()



# %%
from models.lstm_pytorch import LSTMX1_X2BeforeLSTM

set_seed(seed)

n_trials = 50
num_epochs = 500

result_dir = os.path.join(results_dir, f"LSTMX1_X2BeforeLSTM_{n_trials}_trials_{num_epochs}_epochs")
os.makedirs(result_dir, exist_ok=True)  # Create directory if not exists

run_optuna_study(run_cross_val, LSTMX1_X2BeforeLSTM, X1, y, result_dir, n_trials=n_trials, num_epochs=num_epochs, X2=X2)

study = joblib.load(os.path.join(result_dir, "study.pkl"))
study_df = study.trials_dataframe()

print("Validation Losses", study.best_trial.user_attrs["fold_val_losses"])
print("Mean validation loss:", study.best_trial.user_attrs["mean_val_loss"])
print("Test Losses", study.best_trial.user_attrs["test_losses"])
print("Mean test loss:", study.best_trial.user_attrs["mean_test_loss"])
print("Mean test MAE:", study.best_trial.user_attrs["mean_test_mae"])
print("Mean test RMSE:", study.best_trial.user_attrs["mean_test_rmse"])

#get the best model results from study_df and for each fold plot the training loss
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

best_fold_train_losses = []
best_fold_val_losses = []

for i in range(5):
    # Extract losses for the best trial for the current fold
    best_fold_train_losses.append(study_df[f"user_attrs_fold_{i + 1}_train_losses"].iloc[study_df["value"].idxmin()])
    best_fold_val_losses.append(study_df[f"user_attrs_fold_{i + 1}_validation_losses"].iloc[study_df["value"].idxmin()])
    
    # Plot train and validation losses in the current subplot
    axes[i].plot(best_fold_train_losses[i], label="Train Loss")
    axes[i].plot(best_fold_val_losses[i], label="Validation Loss")
    
    # Customize the subplot
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:  # Only set ylabel for the first subplot
        axes[i].set_ylabel("Loss")
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# %%
set_seed(seed)

# Get the test data
epochs_train_losses =[]
epochs_val_losses = []
val_losses = []
test_losses = []
criterion = torch.nn.MSELoss()

best_config = study.best_params

print("Best hyperparameters:", best_config)

#retrain the model 
for fold, (train_idx, test_idx) in enumerate(BlockingTimeSeriesSplit(n_splits=5).split(X1)):
    best_model = LSTMX1_X2BeforeLSTM(input_size=X1.shape[2], hidden_size=best_config["hidden_size"], num_layers=best_config["num_layers"], output_size=1, auxiliary_input_dim=X2.shape[1]).to(device)

    early_stopper = EarlyStopper(patience=10, min_delta=1e-5, min_epochs=100)

    train_val_split_idx = int(0.8 * len(train_idx))
    train_idx, val_index = train_idx[:train_val_split_idx], train_idx[train_val_split_idx:]

    X1_train, X1_val, X1_test, y_train, y_val, y_test = X1[train_idx], X1[val_index], X1[test_idx], y[train_idx], y[val_index], y[test_idx]
    X1_train_scaled, X1_val_scaled, X1_test_scaled = scale_data(X1_train, X1_val, X1_test)
    X2_train, X2_val, X2_test = X2[train_idx], X2[val_index], X2[test_idx]

    train_loader = DataLoader(TensorDataset(X1_train_scaled, X2_train, y_train), batch_size=best_config["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X1_val_scaled, X2_val, y_val), batch_size=len(X1_val_scaled), shuffle=False)
    test_loader = DataLoader(TensorDataset(X1_test_scaled, X2_test, y_test), batch_size=len(X1_test_scaled), shuffle=False)

    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_config["learning_rate"])
    fold_val_loss, best_model, train_losses, validation_losses = train_model_dual_input(best_model, criterion, optimizer, train_loader, val_loader, num_epochs=num_epochs, early_stopper=early_stopper)

    epochs_train_losses.append(train_losses)
    epochs_val_losses.append(validation_losses)
    val_losses.append(fold_val_loss)

    test_loss = 0
    best_model.eval()
    with torch.no_grad():
        for batch_X1, batch_X2, batch_y in test_loader:
            predictions = best_model(batch_X1.to(device), batch_X2.to(device))
            test_loss += criterion(predictions, batch_y.to(device)).item()

    test_losses.append(test_loss / len(test_loader))

print("Validation Losses:", val_losses)
print("Mean validation loss:", np.mean(val_losses))
print("Test Losses:", test_losses)
print("Mean test loss:", np.mean(test_losses))


#plot the train and validation losses for each fold
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

for i in range(5):
    axes[i].plot(epochs_train_losses[i], label="Train Loss")
    axes[i].plot(epochs_val_losses[i], label="Validation Loss")
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:
        axes[i].set_ylabel("Loss")
    axes[i].legend()

plt.tight_layout()
plt.show()


# %%
from models.lstm_pytorch import LSTMX1_X2AfterLSTM

set_seed(seed)

n_trials = 50
num_epochs = 500

result_dir = os.path.join(results_dir, f"LSTMX1_X2AfterLSTM_{n_trials}_trials_{num_epochs}_epochs")
os.makedirs(result_dir, exist_ok=True)  # Create directory if not exists

run_optuna_study(run_cross_val, LSTMX1_X2AfterLSTM, X1, y, result_dir, n_trials=n_trials, num_epochs=num_epochs, X2=X2)

study = joblib.load(os.path.join(result_dir, "study.pkl"))
study_df = study.trials_dataframe()

print("Validation Losses", study.best_trial.user_attrs["fold_val_losses"])
print("Mean validation loss:", study.best_trial.user_attrs["mean_val_loss"])
print("Test Losses", study.best_trial.user_attrs["test_losses"])
print("Mean test loss:", study.best_trial.user_attrs["mean_test_loss"])
print("Mean test MAE:", study.best_trial.user_attrs["mean_test_mae"])
print("Mean test RMSE:", study.best_trial.user_attrs["mean_test_rmse"])

#get the best model results from study_df and for each fold plot the training loss
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

best_fold_train_losses = []
best_fold_val_losses = []

for i in range(5):
    # Extract losses for the best trial for the current fold
    best_fold_train_losses.append(study_df[f"user_attrs_fold_{i + 1}_train_losses"].iloc[study_df["value"].idxmin()])
    best_fold_val_losses.append(study_df[f"user_attrs_fold_{i + 1}_validation_losses"].iloc[study_df["value"].idxmin()])
    
    # Plot train and validation losses in the current subplot
    axes[i].plot(best_fold_train_losses[i], label="Train Loss")
    axes[i].plot(best_fold_val_losses[i], label="Validation Loss")
    
    # Customize the subplot
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:  # Only set ylabel for the first subplot
        axes[i].set_ylabel("Loss")
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# %%
from models.lstm_pytorch import LSTMX1_X2AfterLSTM

set_seed(seed)

n_trials = 50
num_epochs = 500

result_dir = os.path.join(results_dir, f"LSTMX1_X2AfterLSTM_{n_trials}_trials_{num_epochs}_epochs")
os.makedirs(result_dir, exist_ok=True)  # Create directory if not exists

run_optuna_study(run_cross_val, LSTMX1_X2AfterLSTM, X1, y, result_dir, n_trials=n_trials, num_epochs=num_epochs, X2=X2)

study = joblib.load(os.path.join(result_dir, "study.pkl"))
study_df = study.trials_dataframe()

print("Validation Losses", study.best_trial.user_attrs["fold_val_losses"])
print("Mean validation loss:", study.best_trial.user_attrs["mean_val_loss"])
print("Test Losses", study.best_trial.user_attrs["test_losses"])
print("Mean test loss:", study.best_trial.user_attrs["mean_test_loss"])
print("Mean test MAE:", study.best_trial.user_attrs["mean_test_mae"])
print("Mean test RMSE:", study.best_trial.user_attrs["mean_test_rmse"])

#get the best model results from study_df and for each fold plot the training loss
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

best_fold_train_losses = []
best_fold_val_losses = []

for i in range(5):
    # Extract losses for the best trial for the current fold
    best_fold_train_losses.append(study_df[f"user_attrs_fold_{i + 1}_train_losses"].iloc[study_df["value"].idxmin()])
    best_fold_val_losses.append(study_df[f"user_attrs_fold_{i + 1}_validation_losses"].iloc[study_df["value"].idxmin()])
    
    # Plot train and validation losses in the current subplot
    axes[i].plot(best_fold_train_losses[i], label="Train Loss")
    axes[i].plot(best_fold_val_losses[i], label="Validation Loss")
    
    # Customize the subplot
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:  # Only set ylabel for the first subplot
        axes[i].set_ylabel("Loss")
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


# %%
from models.lstm_pytorch import LSTMX1_X2Masking
set_seed(seed)

n_trials = 50
num_epochs = 500

#X1 shape is (num_samples, lookback_period)
masking_X1 = np.zeros((X1.shape[0], X1.shape[1])) 

for i, obs_motif_indexes in enumerate(X2):
    for j, idx in enumerate(obs_motif_indexes):
        masking_X1[i, idx.item():idx.item()+p] = 1

masking_X1 = torch.tensor(masking_X1, dtype=torch.float32)

result_dir = os.path.join(results_dir, f"LSTMX1_X2Masking_{n_trials}_trials_{num_epochs}_epochs")
os.makedirs(result_dir, exist_ok=True)  # Create directory if not exists

run_optuna_study(run_cross_val, LSTMX1_X2Masking, X1, y, result_dir, n_trials=n_trials, num_epochs=num_epochs, X2=masking_X1)

study = joblib.load(os.path.join(result_dir, "study.pkl"))
study_df = study.trials_dataframe()


print("Validation Losses", study.best_trial.user_attrs["fold_val_losses"])
print("Mean validation loss:", study.best_trial.user_attrs["mean_val_loss"])
print("Test Losses", study.best_trial.user_attrs["test_losses"])
print("Mean test loss:", study.best_trial.user_attrs["mean_test_loss"])
print("Mean test MAE:", study.best_trial.user_attrs["mean_test_mae"])
print("Mean test RMSE:", study.best_trial.user_attrs["mean_test_rmse"])

#get the best model results from study_df and for each fold plot the training loss
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

best_fold_train_losses = []
best_fold_val_losses = []

for i in range(5):
    # Extract losses for the best trial for the current fold
    best_fold_train_losses.append(study_df[f"user_attrs_fold_{i + 1}_train_losses"].iloc[study_df["value"].idxmin()])
    best_fold_val_losses.append(study_df[f"user_attrs_fold_{i + 1}_validation_losses"].iloc[study_df["value"].idxmin()])
    
    # Plot train and validation losses in the current subplot
    axes[i].plot(best_fold_train_losses[i], label="Train Loss")
    axes[i].plot(best_fold_val_losses[i], label="Validation Loss")
    
    # Customize the subplot
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:  # Only set ylabel for the first subplot
        axes[i].set_ylabel("Loss")
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()



# %%
from models.lstm_pytorch import LSTMX1Attention

set_seed(seed)

n_trials = 50
num_epochs = 500

result_dir = os.path.join(results_dir, f"LSTMX1Attention_{n_trials}_trials_{num_epochs}_epochs")
os.makedirs(result_dir, exist_ok=True)  # Create directory if not exists

run_optuna_study(run_cross_val, LSTMX1Attention, X1, y, result_dir, n_trials=n_trials, num_epochs=num_epochs)

study = joblib.load(os.path.join(result_dir, "study.pkl"))
study_df = study.trials_dataframe()

print("Validation Losses", study.best_trial.user_attrs["fold_val_losses"])
print("Mean validation loss:", study.best_trial.user_attrs["mean_val_loss"])
print("Test Losses", study.best_trial.user_attrs["test_losses"]) 
print("Mean test loss:", study.best_trial.user_attrs["mean_test_loss"])
print("Mean test MAE:", study.best_trial.user_attrs["mean_test_mae"])
print("Mean test RMSE:", study.best_trial.user_attrs["mean_test_rmse"])

#get the best model results from study_df and for each fold plot the training loss
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 5), sharey=True)

best_fold_train_losses = []
best_fold_val_losses = []

for i in range(5):
    # Extract losses for the best trial for the current fold
    best_fold_train_losses.append(study_df[f"user_attrs_fold_{i + 1}_train_losses"].iloc[study_df["value"].idxmin()])
    best_fold_val_losses.append(study_df[f"user_attrs_fold_{i + 1}_validation_losses"].iloc[study_df["value"].idxmin()])
    
    # Plot train and validation losses in the current subplot
    axes[i].plot(best_fold_train_losses[i], label="Train Loss")
    axes[i].plot(best_fold_val_losses[i], label="Validation Loss")
    
    # Customize the subplot
    axes[i].set_title(f"Fold {i + 1}")
    axes[i].set_xlabel("Epoch")
    if i == 0:  # Only set ylabel for the first subplot
        axes[i].set_ylabel("Loss")
    axes[i].legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()



