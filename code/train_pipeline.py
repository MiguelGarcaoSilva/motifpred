import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from timeseries_split import BlockingTimeSeriesSplit
import numpy as np
import optuna
import joblib
import os
from typing import Tuple, List


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.min_epochs = min_epochs

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ModelTrainingPipeline:
    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def evaluate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        mae = torch.mean(torch.abs(predictions - targets)).item()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
        return mae, rmse

    @staticmethod
    def scale_data(X_train, X_val, X_test):
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = torch.tensor(scaler.fit_transform(X_train.view(-1, X_train.shape[-1])), dtype=torch.float32).view(X_train.shape)
        X_val = torch.tensor(scaler.transform(X_val.view(-1, X_val.shape[-1])), dtype=torch.float32).view(X_val.shape)
        X_test = torch.tensor(scaler.transform(X_test.view(-1, X_test.shape[-1])), dtype=torch.float32).view(X_test.shape)
        return X_train, X_val, X_test

    def train_model(self, model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, early_stopper=None, dual_input=False):
        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        validation_losses = []

        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                if dual_input:
                    batch_X1, batch_X2, batch_y = (t.to(self.device) for t in batch)
                    predictions = model(batch_X1, batch_X2)
                else:
                    batch_X1, batch_y = (t.to(self.device) for t in batch)
                    predictions = model(batch_X1)

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
                    if dual_input:
                        batch_X1, batch_X2, batch_y = (t.to(self.device) for t in batch)
                        predictions = model(batch_X1, batch_X2)
                    else:
                        batch_X1, batch_y = (t.to(self.device) for t in batch)
                        predictions = model(batch_X1)

                    val_loss += criterion(predictions, batch_y).item()

            avg_val_loss = val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)

            # Early stopping
            if early_stopper and epoch >= early_stopper.min_epochs and early_stopper.early_stop(avg_val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        model.load_state_dict(best_model_state)
        return best_val_loss, model, train_losses, validation_losses

    def run_cross_val(self, trial, seed, results_folder, model_class, X1, y, X2=None, criterion=torch.nn.MSELoss(), num_epochs=500):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        fold_results, test_mae_per_fold, test_rmse_per_fold, test_losses = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(BlockingTimeSeriesSplit(n_splits=5).split(X1)):
            early_stopper = EarlyStopper(patience=10, min_delta=1e-5, min_epochs=100)

            # Data split
            train_val_split_idx = int(0.8 * len(train_idx))
            train_idx, val_index = train_idx[:train_val_split_idx], train_idx[train_val_split_idx:]

            X1_train, X1_val, X1_test = X1[train_idx], X1[val_index], X1[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_index], y[test_idx]
            X1_train_scaled, X1_val_scaled, X1_test_scaled = self.scale_data(X1_train, X1_val, X1_test)

            if X2 is not None:
                # Dual input
                X2_train, X2_val, X2_test = X2[train_idx], X2[val_index], X2[test_idx]
                train_loader = DataLoader(TensorDataset(X1_train_scaled, X2_train, y_train), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(X1_val_scaled, X2_val, y_val), batch_size=len(X1_val_scaled), shuffle=False)
                test_loader = DataLoader(TensorDataset(X1_test_scaled, X2_test, y_test), batch_size=len(X1_test_scaled), shuffle=False)

                model = model_class(input_size=X1.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1, auxiliary_input_dim=X2.shape[1]).to(self.device)
                fold_val_loss, model, train_losses, validation_losses = self.train_model(model, criterion, torch.optim.Adam(model.parameters(), lr=learning_rate), train_loader, val_loader, num_epochs, early_stopper, dual_input=True)
            else:
                # Single input
                train_loader = DataLoader(TensorDataset(X1_train_scaled, y_train), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(X1_val_scaled, y_val), batch_size=len(X1_val_scaled), shuffle=False)
                test_loader = DataLoader(TensorDataset(X1_test_scaled, y_test), batch_size=len(X1_test_scaled), shuffle=False)

                model = model_class(input_size=X1.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(self.device)
                fold_val_loss, model, train_losses, validation_losses = self.train_model(model, criterion, torch.optim.Adam(model.parameters(), lr=learning_rate), train_loader, val_loader, num_epochs, early_stopper)

            fold_results.append(fold_val_loss)
            trial.set_user_attr(f"fold_{fold + 1}_train_losses", train_losses)
            trial.set_user_attr(f"fold_{fold + 1}_validation_losses", validation_losses)

        return np.mean(fold_results), model
