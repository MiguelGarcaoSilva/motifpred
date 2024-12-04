import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from timeseries_split import BlockingTimeSeriesSplit
import numpy as np
import os
import random
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
    
    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')


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
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = torch.tensor(scaler.fit_transform(X_train.view(-1, X_train.shape[-1])), dtype=torch.float32).view(X_train.shape)
        X_val = torch.tensor(scaler.transform(X_val.view(-1, X_val.shape[-1])), dtype=torch.float32).view(X_val.shape)
        X_test = torch.tensor(scaler.transform(X_test.view(-1, X_test.shape[-1])), dtype=torch.float32).view(X_test.shape)
        return X_train, X_val, X_test

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


    def prepare_dataloaders(self, X1_train, X1_val, X1_test, y_train, y_val, y_test, batch_size, X2_train=None, X2_val=None, X2_test=None):
        if X2_train is not None:
            train_loader = DataLoader(TensorDataset(X1_train, X2_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X1_val, X2_val, y_val), batch_size=len(X1_val), shuffle=False)
            test_loader = DataLoader(TensorDataset(X1_test, X2_test, y_test), batch_size=len(X1_test), shuffle=False)
        else:
            train_loader = DataLoader(TensorDataset(X1_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X1_val, y_val), batch_size=len(X1_val), shuffle=False)
            test_loader = DataLoader(TensorDataset(X1_test, y_test), batch_size=len(X1_test), shuffle=False)

        return train_loader, val_loader, test_loader


    def train_model(self, model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, dual_input=False):
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = num_epochs
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
            if self.early_stopper and epoch >= self.early_stopper.min_epochs and self.early_stopper.early_stop(avg_val_loss):
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

    def evaluate_test_set(self, model, test_loader, criterion, dual_input=False):
        model.eval()
        all_predictions, all_true_values = [], []
        test_loss = 0

        with torch.no_grad():
            for batch_data in test_loader:
                if dual_input:
                    batch_X1, batch_X2, batch_y = batch_data[0].to(self.device), batch_data[1].to(self.device), batch_data[2].to(self.device)
                    predictions = model(batch_X1, batch_X2)
                else:
                    batch_X1, batch_y = batch_data[0].to(self.device), batch_data[1].to(self.device)
                    predictions = model(batch_X1)

                test_loss += criterion(predictions, batch_y).item()
                all_predictions.append(predictions)
                all_true_values.append(batch_y)

        avg_test_loss = test_loss / len(test_loader)
        all_predictions = torch.cat(all_predictions)
        all_true_values = torch.cat(all_true_values)

        return avg_test_loss, all_predictions, all_true_values



    def run_cross_val(self, trial, seed, results_folder, model_class, model_type, X1, y, X2=None, 
                    criterion=torch.nn.MSELoss(), num_epochs=500, hyperparams=None, model_params_keys=None):
        self.set_seed(seed)
        fold_results, best_epochs, test_losses, test_mae_per_fold, test_rmse_per_fold = [], [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(BlockingTimeSeriesSplit(n_splits=5).split(X1)):
            self.early_stopper.reset()

            # Split and scale data
            train_val_split_idx = int(0.8 * len(train_idx))
            train_idx, val_index = train_idx[:train_val_split_idx], train_idx[train_val_split_idx:]
            X1_train, X1_val, X1_test = X1[train_idx], X1[val_index], X1[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_index], y[test_idx]
            X1_train_scaled, X1_val_scaled, X1_test_scaled = self.scale_data(X1_train, X1_val, X1_test)

            if X2 is not None:
                X2_train, X2_val, X2_test = X2[train_idx], X2[val_index], X2[test_idx]
            else:
                X2_train = X2_val = X2_test = None

            # Prepare DataLoaders
            train_loader, val_loader, test_loader = self.prepare_dataloaders(
                X1_train_scaled, X1_val_scaled, X1_test_scaled, y_train, y_val, y_test, 
                hyperparams['batch_size'], X2_train, X2_val, X2_test
            )

            model_hyperparams = {k: v for k, v in hyperparams.items() if k in model_params_keys}

            
            if model_type == 'LSTM':
                if X2 is not None:
                    model = model_class(input_dim=X1.shape[2],  **model_hyperparams, output_dim=1, auxiliary_input_dim=X2.shape[1]).to(self.device)
                else:
                    model = model_class(input_dim=X1.shape[2], **model_hyperparams, output_dim=1).to(self.device)
            elif model_type == 'FFNN':
                input_dim = X1.shape[2] * X1.shape[1] # Flatten the time series
                if X2 is not None:
                    model = model_class(input_dim=input_dim + X2.shape[1], **model_hyperparams, output_dim=1).to(self.device)
                else:
                    model = model_class(input_dim=input_dim, **model_hyperparams, output_dim=1).to(self.device)

            # Train Model
            fold_val_loss, model, best_epoch, train_losses, validation_losses = self.train_model(
                model, criterion, torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate']),
                train_loader, val_loader, num_epochs, dual_input=(X2 is not None)
            )

            fold_results.append(fold_val_loss)
            trial.set_user_attr(f"fold_{fold + 1}_train_losses", train_losses)
            trial.set_user_attr(f"fold_{fold + 1}_validation_losses", validation_losses)
            best_epochs.append(best_epoch)

            # Test evaluation
            avg_test_loss, fold_predictions, fold_true_values = self.evaluate_test_set(
                model, test_loader, criterion, dual_input=(X2 is not None)
            )
            test_losses.append(avg_test_loss)

            mae, rmse = self.evaluate_metrics(fold_predictions, fold_true_values)
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

