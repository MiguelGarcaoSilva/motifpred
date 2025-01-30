import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from utils.timeseries_split import BlockingTimeSeriesSplit, TrainValTestSplit
import numpy as np
import os
import random
import joblib
import optuna
import time
import math
from typing import Tuple, List


def extract_hyperparameters(trial, suggestion_dict, model_type=None):
    """
    Extracts hyperparameters from a trial object based on the suggestion dictionary and dynamically handles
    layer-specific sampling for different model types.

    Args:
        trial (optuna.Trial): The current Optuna trial.
        suggestion_dict (dict): Dictionary containing parameter definitions.
        model_type (str, optional): Type of the model to handle dynamic parameter extraction.

    Returns:
        dict: Extracted hyperparameters.
    """
    # Extract general hyperparameters
    hyperparameters = {
        param_name: getattr(trial, f"suggest_{param_details['type']}")
        (
            param_name,
            *param_details['args'],  # Pass positional arguments
            **param_details.get('kwargs', {})  # Pass optional keyword arguments
        )
        for param_name, param_details in suggestion_dict.items()
    }

    # Handle FFNN-specific logic
    if model_type == "FFNN":
        num_layers = hyperparameters.pop("num_layers", None)
        hidden_sizes_to_sample = [16, 32, 64, 128, 256]
        hyperparameters["hidden_sizes_list"] = [
            trial.suggest_categorical(f"hidden_size_layer_{i}", hidden_sizes_to_sample)
            for i in range(num_layers)
        ]

    # Handle LSTM-specific logic
    elif model_type == "LSTM":
        num_layers = hyperparameters.pop("num_layers", None)
        hidden_sizes_to_sample = [16, 32, 64, 128, 256]
        hyperparameters["hidden_sizes_list"] = [
            trial.suggest_categorical(f"hidden_size_layer_{i}", hidden_sizes_to_sample)
            for i in range(num_layers)
        ]

    # Handle CNN-specific logic
    elif model_type == "CNN":
        num_layers = hyperparameters.pop("num_layers", None)
        num_filters_to_sample = [16, 32, 64]
        hyperparameters["num_filters_list"] = [
            trial.suggest_categorical(f"num_filters_layer_{i}", num_filters_to_sample)
            for i in range(num_layers)
        ]

    # Handle TCN-specific logic
    elif model_type == "TCN":
        kernel_size = hyperparameters.get("kernel_size", None)
        receptive_field = hyperparameters.pop("receptive_field", None) 
        #get min num of blocks for receptive field
        num_blocks = math.ceil(math.log2((receptive_field - 1) / (kernel_size - 1) + 1))
        num_channels_to_sample = suggestion_dict.get("num_channels_to_sample", {}).get("args", [16, 32])
        hyperparameters["num_channels_list"] = [
            trial.suggest_categorical(f"block_channels_{i}", num_channels_to_sample)
            for i in range(num_blocks)
        ]
    elif model_type == 'Baseline':
        return hyperparameters


    return hyperparameters




def run_optuna_study(objective_func, model_class, model_type, suggestion_dict, model_params_keys, seed, X, y, normalize_flags, results_folder: str, n_trials: int = 100, num_epochs=500):

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    file_name = os.path.join(results_folder, "study.pkl")
    
    def objective(trial):
        # Extract basic hyperparameters from the suggestion dictionary
        hyperparameters = extract_hyperparameters(trial, suggestion_dict, model_type=model_type)
            
        criterion = torch.nn.MSELoss()  # Define the criterion here
        trial_val_loss, _, _ = objective_func(trial, seed, results_folder, model_class, model_type, X, y, normalize_flags, criterion, num_epochs, hyperparameters, model_params_keys)  # Pass hyperparameters

        return trial_val_loss

    # Let Optuna manage trials and pass them to the objective function
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    joblib.dump(study, file_name)

    # Save and log the study results
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(results_folder, "study_results.csv"), index=False)

    print("Best hyperparameters:", study.best_params)


def get_preds_best_config_train_val_test(study, pipeline, model_class, model_type, model_params_keys, num_epochs, seed, X, y, normalize_flags):
    pipeline.set_seed(seed)

    # Retrieve the best configuration from the Optuna study
    best_config = study.best_params
    print("Best hyperparameters:", best_config)

    # Initialize lists to store results
    epochs_train_losses, epochs_val_losses, val_losses, test_losses, test_mae, test_rmse, all_predictions, all_true_values = [], [], [], [], [], [], [], []

    # Extract input data from the dictionary, with None defaults
    X_series, X_mask, X_indices = X.get('X_series'), X.get('X_mask'), X.get('X_indices')
    
    # Initialize the splitter
    splitter = TrainValTestSplit(val_size=0.15, test_size=0.15)

    # Split data into train, val, and test sets
    if X_series is not None:
        train_indices, val_indices, test_indices = splitter.split(X_series)
        X_train, X_val, X_test = X_series[train_indices], X_series[val_indices], X_series[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    elif X_indices is not None:
        train_indices, val_indices, test_indices = splitter.split(X_indices)
        X_train, X_val, X_test = X_indices[train_indices], X_indices[val_indices], X_indices[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    else:
        raise ValueError("No valid input data found in X dictionary.")

    # Process each input based on normalization flags
    X1_train = X1_val = X1_test = None
    X2_train = X2_val = X2_test = None
    X3_train = X3_val = X3_test = None

    if X_series is not None and X_mask is None and X_indices is None:
        if normalize_flags['X_series']:
            X1_train, X1_val, X1_test = pipeline.scale_data(X_train, X_val, X_test)
    elif X_series is not None and X_mask is not None and X_indices is None:
        X1_train, X1_val, X1_test = X_train, X_val, X_test
        X2_train, X2_val, X2_test = X_mask[train_indices], X_mask[val_indices], X_mask[test_indices]
        if normalize_flags['X_series']:
            X1_train, X1_val, X1_test = pipeline.scale_data(X1_train, X1_val, X1_test)
        if normalize_flags['X_mask']:
            raise ValueError("Masking should not be normalized.")
    elif X_series is None and X_mask is None and X_indices is not None:
        X3_train, X3_val, X3_test = X_train, X_val, X_test
        if normalize_flags['X_indices']:
            X3_train, X3_val, X3_test = pipeline.scale_data(X3_train, X3_val, X3_test)
    elif X_series is not None and X_mask is not None and X_indices is not None:
        X1_train, X1_val, X1_test = X_train, X_val, X_test
        X2_train, X2_val, X2_test = X_mask[train_indices], X_mask[val_indices], X_mask[test_indices]
        X3_train, X3_val, X3_test = X_indices[train_indices], X_indices[val_indices], X_indices[test_indices]
        if normalize_flags['X_series']:
            X1_train, X1_val, X1_test = pipeline.scale_data(X1_train, X1_val, X1_test)
        if normalize_flags['X_mask']:
            raise ValueError("Masking should not be normalized.")
        if normalize_flags['X_indices']:
            X3_train, X3_val, X3_test = pipeline.scale_data(X3_train, X3_val, X3_test)

    # Prepare DataLoaders
    X_train, input_dim = pipeline.prepare_input_data(model_type, series=X1_train, mask=X2_train, indices=X3_train)
    X_val, _ = pipeline.prepare_input_data(model_type, series=X1_val, mask=X2_val, indices=X3_val)
    X_test, _ = pipeline.prepare_input_data(model_type, series=X1_test, mask=X2_test, indices=X3_test)
    train_loader, val_loader, test_loader = pipeline.prepare_dataloaders(
        X_train, X_val, X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        batch_size=best_config['batch_size']
    )

    # Initialize model
    model_hyperparams = {k: v for k, v in best_config.items() if k in model_params_keys}

    if model_type == 'FFNN':
        model_hyperparams["hidden_sizes_list"] = [best_config[f"hidden_size_layer_{layer}"] for layer in range(best_config["num_layers"])] 
        model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(pipeline.device)
    elif model_type == 'LSTM':
        model_hyperparams["hidden_sizes_list"] = [best_config[f"hidden_size_layer_{layer}"] for layer in range(best_config["num_layers"])] 
        model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(pipeline.device)
    elif model_type == 'CNN':
        input_channels = input_dim
        model_hyperparams["num_filters_list"] = [best_config[f"num_filters_layer_{layer}"] for layer in range(best_config["num_layers"])]
        model = model_class(input_channels=input_channels, sequence_length=X_train.shape[1], output_dim=1, **model_hyperparams).to(pipeline.device)
    elif model_type == 'TCN':
        input_channels = input_dim
        num_blocks = len([key for key in best_config.keys() if 'block_channels_' in key])
        model_hyperparams["num_channels_list"] = [best_config[f"block_channels_{layer}"] for layer in range(num_blocks)]
        model = model_class(input_channels=input_channels, output_dim=1, **model_hyperparams).to(pipeline.device)
    elif model_type == 'Transformer':
        model = model_class(input_dim=input_dim, sequence_length=X_train.shape[1], output_dim=1, **model_hyperparams).to(pipeline.device)
    elif model_type == 'Baseline':
        model = model_class(n_timepoints=input_dim).to(pipeline.device)

    if model_type == 'Baseline':
        val_loss, model, best_epoch, train_losses, validation_losses = float('inf'), model, 0, [], []
    else:
        val_loss, model, best_epoch, train_losses, validation_losses = pipeline.train_model(
            model, criterion=torch.nn.MSELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=best_config['learning_rate']),
            train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs
        )

    # Store training and validation losses
    epochs_train_losses = train_losses
    epochs_val_losses = validation_losses
    val_losses = val_loss

    # Evaluate the model on the test set
    test_loss, test_predictions, test_true_values = pipeline.evaluate_test_set(
        model, test_loader, criterion=torch.nn.MSELoss()
    )

    test_losses = test_loss
    mae, rmse = pipeline.evaluate_metrics(test_predictions, test_true_values)
    test_mae = mae
    test_rmse = rmse
    all_predictions = test_predictions.cpu().numpy()
    all_true_values = test_true_values.cpu().numpy()
    
    return epochs_train_losses, epochs_val_losses, val_losses, test_losses, test_mae, test_rmse, all_predictions, all_true_values


def get_preds_best_config(study, pipeline, model_class, model_type, model_params_keys, num_epochs, seed, X, y, normalize_flags):
    pipeline.set_seed(seed)

    # Retrieve the best configuration from the Optuna study
    best_config = study.best_params
    print("Best hyperparameters:", best_config)

    # Initialize lists to store results
    epochs_train_losses, epochs_val_losses, val_losses, test_losses, test_mae_per_fold, test_rmse_per_fold, all_predictions, all_true_values = [], [], [], [], [], [], [], []

    # Extract input data from the dictionary, with None defaults
    X_series, X_mask, X_indices = X.get('X_series'), X.get('X_mask'), X.get('X_indices')
    
    data_indices = X_series if X_series is not None else X_indices

    for fold, (train_idx, test_idx) in enumerate(BlockingTimeSeriesSplit(n_splits=5).split(data_indices)):
        pipeline.early_stopper.reset()

        # Split indices for train, validation, and test
        train_val_split_idx = int(0.8 * len(train_idx))
        train_idx, val_idx = train_idx[:train_val_split_idx], train_idx[train_val_split_idx:]

        # Process each input based on normalization flags
        X1_train = X1_val = X1_test = None
        X2_train = X2_val = X2_test = None
        X3_train = X3_val = X3_test = None

        if X_series is not None and X_mask is None and X_indices is None:
            X1_train, X1_val, X1_test = X_series[train_idx], X_series[val_idx], X_series[test_idx]
            if normalize_flags['X_series']:
                X1_train, X1_val, X1_test = pipeline.scale_data(X1_train, X1_val, X1_test)

        elif X_series is not None and X_mask is not None and X_indices is None:
            X1_train, X1_val, X1_test = X_series[train_idx], X_series[val_idx], X_series[test_idx]
            X2_train, X2_val, X2_test = X_mask[train_idx], X_mask[val_idx], X_mask[test_idx]
            if normalize_flags['X_series']:
                X1_train, X1_val, X1_test = pipeline.scale_data(X1_train, X1_val, X1_test)
            if normalize_flags['X_mask']:
                raise ValueError("Masking should not be normalized.")

        elif X_series is None and X_mask is None and X_indices is not None:
            X3_train, X3_val, X3_test = X_indices[train_idx], X_indices[val_idx], X_indices[test_idx]
            if normalize_flags['X_indices']:
                X3_train, X3_val, X3_test = pipeline.scale_data(X3_train, X3_val, X3_test)

        elif X_series is not None and X_mask is not None and X_indices is not None:
            X1_train, X1_val, X1_test = X_series[train_idx], X_series[val_idx], X_series[test_idx]
            X2_train, X2_val, X2_test = X_mask[train_idx], X_mask[val_idx], X_mask[test_idx]
            X3_train, X3_val, X3_test = X_indices[train_idx], X_indices[val_idx], X_indices[test_idx]
            if normalize_flags['X_series']:
                X1_train, X1_val, X1_test = pipeline.scale_data(X1_train, X1_val, X1_test)
            if normalize_flags['X_mask']:
                raise ValueError("Masking should not be normalized.")
            if normalize_flags['X_indices']:
                X3_train, X3_val, X3_test = pipeline.scale_data(X3_train, X3_val, X3_test)

        # Prepare DataLoaders
        X_train, input_dim = pipeline.prepare_input_data(model_type, series=X1_train, mask=X2_train, indices=X3_train)
        X_val, _ = pipeline.prepare_input_data(model_type, series=X1_val, mask=X2_val, indices=X3_val)
        X_test, _ = pipeline.prepare_input_data(model_type, series=X1_test, mask=X2_test, indices=X3_test)
        train_loader, val_loader, test_loader = pipeline.prepare_dataloaders(
            X_train, X_val, X_test,
            y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
            batch_size=best_config['batch_size']
        )

        model_hyperparams = {k: v for k, v in best_config.items() if k in model_params_keys}

        if model_type == 'FFNN':
            model_hyperparams["hidden_sizes_list"] = [best_config[f"hidden_size_layer_{layer}"] for layer in range(best_config["num_layers"])] 
            model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(pipeline.device)
        elif model_type == 'LSTM':
            model_hyperparams["hidden_sizes_list"] = [best_config[f"hidden_size_layer_{layer}"] for layer in range(best_config["num_layers"])] 
            model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(pipeline.device)
        elif model_type == 'CNN':
            input_channels = input_dim
            model_hyperparams["num_filters_list"] = [best_config[f"num_filters_layer_{layer}"] for layer in range(best_config["num_layers"])]
            model = model_class(input_channels=input_channels, sequence_length=X_train.shape[1], output_dim=1, **model_hyperparams).to(pipeline.device)
        elif model_type == 'TCN':
            input_channels = input_dim
            num_blocks = len([key for key in best_config.keys() if 'block_channels_' in key])
            model_hyperparams["num_channels_list"] = [best_config[f"block_channels_{layer}"] for layer in range(num_blocks)]
            model = model_class(input_channels=input_channels, output_dim=1, **model_hyperparams).to(pipeline.device)
        elif model_type == 'Transformer':
            model = model_class(input_dim=input_dim, sequence_length=X_train.shape[1], output_dim=1, **model_hyperparams).to(pipeline.device)
        elif model_type == 'Baseline':
            model = model_class(n_timepoints = input_dim).to(pipeline.device)

        if model_type == 'Baseline':
            fold_val_loss, model, best_epoch, train_losses, validation_losses = float('inf'), model, 0, [], []
        else:
            fold_val_loss, model, best_epoch, train_losses, validation_losses = pipeline.train_model(
                model, criterion=torch.nn.MSELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=best_config['learning_rate']),
                train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs
            )

        # Store training and validation losses
        epochs_train_losses.append(train_losses)
        epochs_val_losses.append(validation_losses)
        val_losses.append(fold_val_loss)

        # Evaluate the model on the test set
        test_loss, fold_predictions, fold_true_values = pipeline.evaluate_test_set(
            model, test_loader, criterion=torch.nn.MSELoss()
        )

        test_losses.append(test_loss)
        mae, rmse = pipeline.evaluate_metrics(fold_predictions, fold_true_values)
        test_mae_per_fold.append(mae)
        test_rmse_per_fold.append(rmse)
        all_predictions.append(fold_predictions.cpu().numpy())
        all_true_values.append(fold_true_values.cpu().numpy())

    # Output validation and test losses
    print("Validation Losses:", val_losses)
    print("Mean validation loss:", np.mean(val_losses))
    print("Test Losses:", test_losses)
    print("Mean test loss:", np.mean(test_losses))
    print("Test MAE:", test_mae_per_fold)
    
    return epochs_train_losses, epochs_val_losses, val_losses, test_losses, test_mae_per_fold, test_rmse_per_fold, all_predictions, all_true_values




class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, min_epochs=100, max_time_minutes=8):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.min_epochs = min_epochs
        self.start_time = None
        self.max_time_seconds = max_time_minutes * 60

    def start_timer(self):
        """Initialize the timer at the start of training."""
        self.start_time = time.time()

    def has_time_exceeded(self):
        """Check if the training time has exceeded the maximum allowed time."""
        if self.start_time is None:
            raise ValueError("Timer not started. Call `start_timer()` at the beginning of training.")
        elapsed_time = time.time() - self.start_time
        return elapsed_time > self.max_time_seconds

    def early_stop(self, validation_loss, current_epoch):
        """Determine if training should stop early."""
        # Check if training exceeds minimum required epochs
        if current_epoch < self.min_epochs:
            return False
        
        # Check if validation loss improved
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        # Check if time limit has been exceeded
        if self.has_time_exceeded():
            print("Stopping early: Maximum training time exceeded.")
            return True

        return False

    def reset(self):
        """Reset the early stopper's state."""
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.start_time = None



class ModelTrainingPipeline:
    def __init__(self, device: torch.device, early_stopper=None):
        self.device = device
        self.early_stopper = early_stopper

    @staticmethod
    def evaluate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        mae = torch.mean(torch.abs(predictions - targets)).item()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
        return mae, rmse

    @staticmethod
    def scale_data(X_train, X_val, X_test):
        def scale_subset(X, scaler):
            # Create a mask for unmasked values (not equal to -1)
            mask = X != -1
            # Flatten the data and apply the mask to select unmasked values
            unmasked_values = X[mask].view(-1, 1).numpy()
            # Scale only the unmasked values
            scaled_values = scaler.transform(unmasked_values)
            # Create a copy of the original data and replace unmasked values with scaled values
            scaled_X = X.clone()
            scaled_X[mask] = torch.tensor(scaled_values.flatten(), dtype=torch.float32)
            return scaled_X

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_mask = X_train != -1
        train_values = X_train[train_mask].view(-1, 1).numpy()
        scaler.fit(train_values)
        X_train = scale_subset(X_train, scaler)
        X_val = scale_subset(X_val, scaler)
        X_test = scale_subset(X_test, scaler)
        
        return X_train, X_val, X_test
    



    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


    def prepare_dataloaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size):

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=len(X_val), shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=len(X_test), shuffle=False)

        return train_loader, val_loader, test_loader


    def prepare_input_data(self, model_type, series=None, mask=None, indices=None):
        input_flags = {
            "series": series is not None,
            "mask": mask is not None,
            "indices": indices is not None
        }

        if model_type == "FFNN":
            if input_flags["series"] and not input_flags["mask"] and not input_flags["indices"]:
                X = series.view(series.size(0), -1)
            elif input_flags["series"] and input_flags["mask"] and not input_flags["indices"]:
                X = torch.cat((series.view(series.size(0), -1), mask), dim=1)
            elif not input_flags["series"] and not input_flags["mask"] and input_flags["indices"]:
                X = indices.squeeze(-1)
            elif input_flags["series"] and input_flags["mask"] and input_flags["indices"]:
                X = torch.cat((series.view(series.size(0), -1), mask, indices), dim=1)
            else:
                raise ValueError("Invalid input data for FFNN model.")

            input_dim = X.size(1)

        elif model_type in {"LSTM", "CNN", "TCN", "Transformer"}:
            if input_flags["series"] and not input_flags["mask"] and not input_flags["indices"]:
                X = series
            elif input_flags["series"] and input_flags["mask"] and not input_flags["indices"]:
                X = torch.cat((series, mask.unsqueeze(-1)), dim=2)
            elif not input_flags["series"] and not input_flags["mask"] and input_flags["indices"]:
                X = indices
            elif input_flags["series"] and input_flags["mask"] and input_flags["indices"]:
                raise ValueError("To implement.")

            input_dim = X.size(2)

        elif model_type in {"Baseline"}:
            X = indices
            input_dim = series.size(1)

        return X , input_dim


    def train_model(self, model, criterion, optimizer, train_loader, val_loader, num_epochs=1000):
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = num_epochs
        train_losses, validation_losses = [], []

        # Start timer for early stopping
        if self.early_stopper:
            self.early_stopper.start_timer()

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                batch = tuple(t.to(self.device) for t in batch)
                input, batch_y = batch
                predictions = model(input)

                optimizer.zero_grad()
                loss = criterion(predictions, batch_y)
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = tuple(t.to(self.device) for t in batch)

                    input, batch_y = batch
                    predictions = model(input)

                    val_loss += criterion(predictions, batch_y).item()

            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)

            # Early stopping
            if self.early_stopper and self.early_stopper.early_stop(avg_val_loss, epoch):
                print(f"Early stopping at epoch {epoch + 1}, with best epoch being {best_epoch}")
                break

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch

            if epoch == num_epochs - 1:
                print(f"Training completed all epochs. Best epoch was {best_epoch}")

        if best_model_state:
            model.load_state_dict(best_model_state)

        return best_val_loss, model, best_epoch, train_losses, validation_losses



    def evaluate_test_set(self, model, test_loader, criterion):
        model.eval()
        all_predictions, all_true_values = [], []
        test_loss = 0

        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = tuple(t.to(self.device) for t in batch_data)
                
                # Dynamically unpack inputs
                *inputs, batch_y = batch_data
                predictions = model(*inputs)

                test_loss += criterion(predictions, batch_y).item()
                all_predictions.append(predictions)
                all_true_values.append(batch_y)

        avg_test_loss = test_loss / len(test_loader)
        all_predictions = torch.cat(all_predictions)
        all_true_values = torch.cat(all_true_values)

        return avg_test_loss, all_predictions, all_true_values


 
    def run_cross_val(self, trial, seed, results_folder, model_class, model_type, X, y, normalize_flags,
                        criterion=torch.nn.MSELoss(), num_epochs=500, hyperparams=None, model_params_keys=None):
            self.set_seed(seed)

            fold_results, best_epochs, test_losses, test_mae_per_fold, test_rmse_per_fold = [], [], [], [], []

            #get X from X dictionary, if doesnt exist, set to None
            X_series, X_mask, X_indices = X.get('X_series'), X.get('X_mask'), X.get('X_indices')
            
            data_indices = X_series if X_series is not None else X_indices

            for fold, (train_idx, test_idx) in enumerate(BlockingTimeSeriesSplit(n_splits=5).split(data_indices)):
                self.early_stopper.reset()

                # Split indices for train, validation, and test
                train_val_split_idx = int(0.8 * len(train_idx))
                train_idx, val_idx = train_idx[:train_val_split_idx], train_idx[train_val_split_idx:]

                # Process each input in X_inputs based on normalization flags
                X1_train = X1_val = X1_test = None
                X2_train = X2_val = X2_test = None
                X3_train = X3_val = X3_test = None

                # if only series
                if X_series is not None and X_mask is None and X_indices is None:
                    X1_train, X1_val, X1_test = X['X_series'][train_idx], X['X_series'][val_idx], X['X_series'][test_idx]
                    if normalize_flags['X_series']:
                        X1_train, X1_val, X1_test = self.scale_data(X1_train, X1_val, X1_test)
                #if series and mask
                elif X_series is not None and X_mask is not None and X_indices is None:
                    X1_train, X1_val, X1_test = X['X_series'][train_idx], X['X_series'][val_idx], X['X_series'][test_idx]
                    X2_train, X2_val, X2_test = X['X_mask'][train_idx], X['X_mask'][val_idx], X['X_mask'][test_idx]
                    if normalize_flags['X_series']:
                        X1_train, X1_val, X1_test = self.scale_data(X1_train, X1_val, X1_test)
                    if normalize_flags['X_mask']:
                        raise ValueError("Masking should not be normalized.")
                #if only indices
                elif X_series is None and X_mask is None and X_indices is not None:
                    X3_train, X3_val, X3_test = X['X_indices'][train_idx], X['X_indices'][val_idx], X['X_indices'][test_idx]
                    if normalize_flags['X_indices']:
                        X3_train, X3_val, X3_test = self.scale_data(X3_train, X3_val, X3_test)
                #if series, mask and indices
                elif X_series is not None and X_mask is not None and X_indices is not None:
                    X1_train, X1_val, X1_test = X['X_series'][train_idx], X['X_series'][val_idx], X['X_series'][test_idx]
                    X2_train, X2_val, X2_test = X['X_mask'][train_idx], X['X_mask'][val_idx], X['X_mask'][test_idx]
                    X3_train, X3_val, X3_test = X['X_indices'][train_idx], X['X_indices'][val_idx], X['X_indices'][test_idx]
                    if normalize_flags['X_series']:
                        X1_train, X1_val, X1_test = self.scale_data(X1_train, X1_val, X1_test)
                    if normalize_flags['X_mask']:
                        raise ValueError("Masking should not be normalized.")
                    if normalize_flags['X_indices']:
                        X3_train, X3_val, X3_test = self.scale_data(X3_train, X3_val, X3_test)

                # Prepare DataLoaders
                X_train, input_dim = self.prepare_input_data(model_type, series=X1_train, mask=X2_train, indices=X3_train)
                X_val, input_dim = self.prepare_input_data(model_type, series=X1_val, mask=X2_val, indices=X3_val)
                X_test, input_dim = self.prepare_input_data(model_type, series=X1_test, mask=X2_test, indices=X3_test)
                train_loader, val_loader, test_loader = self.prepare_dataloaders(
                    X_train, X_val, X_test,
                    y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
                    batch_size=hyperparams['batch_size']
                )

                model_hyperparams = {k: v for k, v in hyperparams.items() if k in model_params_keys}
                        
                # Adjust model input dimensions based on input data
                if model_type == 'FFNN':
                    model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(self.device)
                if model_type == 'LSTM':
                    model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(self.device)
                elif model_type == 'CNN':
                    input_channels = input_dim
                    model = model_class(input_channels=input_channels, sequence_length=X_train.shape[1], **model_hyperparams, output_dim=1,).to(self.device)
                elif model_type == 'TCN':
                    input_channels = input_dim
                    model = model_class(input_channels=input_channels, **model_hyperparams, output_dim=1).to(self.device)
                elif model_type == 'Transformer':
                    model = model_class(input_dim=input_dim, sequence_length=X_train.shape[1], **model_hyperparams, output_dim=1).to(self.device)
                elif model_type == 'Informer':
                    model = model_class(enc_in=input_dim, dec_in=input_dim, c_out=1,  seq_len=X_train.shape[1], 
                                        label_len=int(X_train.shape[1] // 2), **model_hyperparams,  out_len=1, device=self.device).to(self.device)
                elif model_type == 'Baseline':
                    model = model_class(n_timepoints = input_dim).to(self.device)

                if model_type == 'Baseline':  # Skip training for the baseline model
                    fold_val_loss, model, best_epoch, train_losses, validation_losses = float('inf'), model, 0, [], []
                else:
                    # Train the model using the existing train_model method
                    fold_val_loss, model, best_epoch, train_losses, validation_losses = self.train_model(
                        model, criterion, torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate']),
                        train_loader, val_loader, num_epochs
                    )

                # Test evaluation for other models
                fold_test_loss, test_fold_predictions, test_fold_true_values = self.evaluate_test_set(
                    model, test_loader, criterion
                )
                test_losses.append(fold_test_loss)
                mae, rmse = self.evaluate_metrics(test_fold_predictions, test_fold_true_values)
                test_mae_per_fold.append(mae)
                test_rmse_per_fold.append(rmse)

                # Store results for Optuna logging
                fold_results.append(fold_val_loss)
                trial.set_user_attr(f"fold_{fold + 1}_train_losses", train_losses)
                trial.set_user_attr(f"fold_{fold + 1}_validation_losses", validation_losses)
                best_epochs.append(best_epoch)

            # Calculate mean and std metrics across folds
            mean_val_loss = np.mean(fold_results)
            mean_test_loss = np.mean(test_losses)
            mean_test_mae, std_test_mae = np.mean(test_mae_per_fold), np.std(test_mae_per_fold)
            mean_test_rmse, std_test_rmse = np.mean(test_rmse_per_fold), np.std(test_rmse_per_fold)

            # Log metrics to Optuna
            trial.set_user_attr("best_epochs", best_epochs)
            trial.set_user_attr("fold_val_losses", fold_results)
            trial.set_user_attr("mean_val_loss", mean_val_loss)
            trial.set_user_attr("test_losses", test_losses)
            trial.set_user_attr("mean_test_loss", mean_test_loss)
            trial.set_user_attr("test_mae_per_fold", test_mae_per_fold)
            trial.set_user_attr("mean_test_mae", mean_test_mae)
            trial.set_user_attr("std_test_mae", std_test_mae)
            trial.set_user_attr("test_rmse_per_fold", test_rmse_per_fold)
            trial.set_user_attr("mean_test_rmse", mean_test_rmse)
            trial.set_user_attr("std_test_rmse", std_test_rmse)

            return mean_val_loss, mean_test_loss, model

    def run_train_val_test(self, trial, seed, results_folder, model_class, model_type, X, y, normalize_flags,
                        criterion=torch.nn.MSELoss(), num_epochs=500, hyperparams=None, model_params_keys=None):
        """
        Runs a single train-val-test split.
        """
        self.set_seed(seed)

        # Get X from X dictionary, if it doesn't exist, set to None
        X_series, X_mask, X_indices = X.get('X_series'), X.get('X_mask'), X.get('X_indices')
        
        splitter = TrainValTestSplit(val_size=0.15, test_size=0.15)

        # Split data into train, val, and test sets
        if X_series is not None:
            train_indices, val_indices, test_indices = splitter.split(X_series)
            X_train, X_val, X_test = X_series[train_indices], X_series[val_indices], X_series[test_indices]
            y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
        elif X_indices is not None:
            train_indices, val_indices, test_indices = splitter.split(X_indices)
            X_train, X_val, X_test = X_indices[train_indices], X_indices[val_indices], X_indices[test_indices]
            y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
        else:
            raise ValueError("No valid input data found in X dictionary.")

        # Process each input based on normalization flags
        X1_train = X1_val = X1_test = None
        X2_train = X2_val = X2_test = None
        X3_train = X3_val = X3_test = None

        if X_series is not None and X_mask is None and X_indices is None:
            if normalize_flags['X_series']:
                X1_train, X1_val, X1_test = self.scale_data(X_train, X_val, X_test)
        elif X_series is not None and X_mask is not None and X_indices is None:
            X1_train, X1_val, X1_test = X_train, X_val, X_test
            X2_train, X2_val, X2_test = X_mask[train_indices], X_mask[val_indices], X_mask[test_indices]
            if normalize_flags['X_series']:
                X1_train, X1_val, X1_test = self.scale_data(X1_train, X1_val, X1_test)
            if normalize_flags['X_mask']:
                raise ValueError("Masking should not be normalized.")
        elif X_series is None and X_mask is None and X_indices is not None:
            X3_train, X3_val, X3_test = X_train, X_val, X_test
            if normalize_flags['X_indices']:
                X3_train, X3_val, X3_test = self.scale_data(X3_train, X3_val, X3_test)
        elif X_series is not None and X_mask is not None and X_indices is not None:
            X1_train, X1_val, X1_test = X_train, X_val, X_test
            X2_train, X2_val, X2_test = X_mask[train_indices], X_mask[val_indices], X_mask[test_indices]
            X3_train, X3_val, X3_test = X_indices[train_indices], X_indices[val_indices], X_indices[test_indices]
            if normalize_flags['X_series']:
                X1_train, X1_val, X1_test = self.scale_data(X1_train, X1_val, X1_test)
            if normalize_flags['X_mask']:
                raise ValueError("Masking should not be normalized.")
            if normalize_flags['X_indices']:
                X3_train, X3_val, X3_test = self.scale_data(X3_train, X3_val, X3_test)

        # Prepare DataLoaders
        X_train, input_dim = self.prepare_input_data(model_type, series=X1_train, mask=X2_train, indices=X3_train)
        X_val, _ = self.prepare_input_data(model_type, series=X1_val, mask=X2_val, indices=X3_val)
        X_test, _ = self.prepare_input_data(model_type, series=X1_test, mask=X2_test, indices=X3_test)
        train_loader, val_loader, test_loader = self.prepare_dataloaders(
            X_train, X_val, X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            batch_size=hyperparams['batch_size']
        )

        # Initialize model
        model_hyperparams = {k: v for k, v in hyperparams.items() if k in model_params_keys}
        if model_type == 'FFNN':
            model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(self.device)
        elif model_type == 'LSTM':
            model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(self.device)
        elif model_type == 'CNN':
            input_channels = input_dim
            model = model_class(input_channels=input_channels, sequence_length=X_train.shape[1], **model_hyperparams, output_dim=1).to(self.device)
        elif model_type == 'TCN':
            input_channels = input_dim
            model = model_class(input_channels=input_channels, **model_hyperparams, output_dim=1).to(self.device)
        elif model_type == 'Transformer':
            model = model_class(input_dim=input_dim, sequence_length=X_train.shape[1], **model_hyperparams, output_dim=1).to(self.device)
        elif model_type == 'Baseline':
            model = model_class(n_timepoints=input_dim).to(self.device)

        # Train the model
        if model_type == 'Baseline':
            val_loss, model, best_epoch, train_losses, validation_losses = float('inf'), model, 0, [], []
        else:
            val_loss, model, best_epoch, train_losses, validation_losses = self.train_model(
                model, criterion, torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate']),
                train_loader, val_loader, num_epochs
            )

        # Evaluate on the test set
        test_loss, test_predictions, test_true_values = self.evaluate_test_set(model, test_loader, criterion)
        mae, rmse = self.evaluate_metrics(test_predictions, test_true_values)

        # Log results
        trial.set_user_attr("train_losses", train_losses)
        trial.set_user_attr("validation_losses", validation_losses)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("test_loss", test_loss)
        trial.set_user_attr("test_mae", mae)
        trial.set_user_attr("test_rmse", rmse)

        return val_loss, test_loss, model
