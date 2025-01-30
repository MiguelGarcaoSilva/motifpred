from matplotlib import pyplot as plt
import numpy as np
from utils.timeseries_split import BlockingTimeSeriesSplit
from torch.nn.utils.rnn import pad_sequence
import torch
import plotly.graph_objects as go

def create_dataset(data, lookback_period, step, forecast_period, motif_indexes, motif_size):
    X1, X2, mask, y = [], [], [], []  # X1: data, X2: indexes of the motifs, y: distance to the next motif

    for idx in range(len(data[0]) - lookback_period - 1):
        if idx % step != 0:
            continue

        window_end_idx = idx + lookback_period
        forecast_period_end = window_end_idx + forecast_period

        # If there are no more matches after the window, break
        if not any(window_end_idx < motif_idx for motif_idx in motif_indexes):
            break

        # Motif indexes in window, relative to the start of the window
        motif_indexes_in_window = [motif_idx - idx for motif_idx in motif_indexes if idx <= motif_idx <= window_end_idx]
        motif_indexes_in_forecast_period = [motif_idx for motif_idx in motif_indexes if window_end_idx < motif_idx <= forecast_period_end]

        if motif_indexes_in_forecast_period:
            next_match_in_forecast_period = motif_indexes_in_forecast_period[0]
        else:
            continue  # No match in the forecast period but exists in the future

        # Get the data window and transpose to (lookback_period, num_features)
        data_window = data[:, idx:window_end_idx].T

        #mask for the motif
        motif_mask = torch.zeros(lookback_period, dtype=torch.float32)  # Initialize mask with zeros
        motif_indexes_in_window = sorted(motif_indexes_in_window)
        for motif_start in motif_indexes_in_window:
            motif_end = motif_start + motif_size
            if motif_start < lookback_period and motif_end > 0:
                motif_mask[max(0, motif_start):min(lookback_period, motif_end)] = 1

        # Index of the next match relative to the end of the window
        data_y = next_match_in_forecast_period - window_end_idx
        
        # Append to lists
        X1.append(torch.tensor(data_window, dtype=torch.float32))  # Now with shape (lookback_period, num_features)
        X2.append(torch.tensor(motif_indexes_in_window, dtype=torch.float32))
        mask.append(motif_mask)
        y.append(data_y)

    # Pad X2 sequences to have the same length
    X2_padded = pad_sequence(X2, batch_first=True, padding_value=-1).unsqueeze(-1) # Final shape: (num_samples, max_num_motifs, 1)
    # Convert lists to torch tensors
    X1 = torch.stack(X1)  # Final shape: (num_samples, lookback_period, num_features)
    mask = torch.stack(mask)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X1, X2_padded, mask, y


def create_multi_motif_dataset(data, lookback_period, step, forecast_period, motif_indexes_list, motif_sizes_list):
    X1, X2, mask, y = [], [], [], []

    for idx in range(0, len(data[0]) - lookback_period - forecast_period, step):
        window_end_idx = idx + lookback_period
        forecast_period_end = window_end_idx + forecast_period

        # Extract the data window and transpose to (lookback_period, num_features)
        data_window = data[:, idx:window_end_idx].T

        motif_indexes_in_window = []  # Stores motif indices for the lookback window
        mask_windows = []             # Stores masks for each motif in the lookback window
        forecast_distances = []       # Stores forecast distances for each motif

        valid_instance = False

        # for each motif, check if it is in the lookback window and forecast period
        for motif_indexes, motif_size in zip(motif_indexes_list, motif_sizes_list):
            mask_window = torch.zeros(lookback_period, dtype=torch.float32)  # Initialize mask with zeros
            motif_indexes = sorted(motif_indexes)  
            # Motif indexes in the lookback period (relative to the start of the window)
            motif_in_mask = sorted([
                int(motif_idx) - idx
                for motif_idx in motif_indexes
                if (motif_idx + motif_size > idx and motif_idx < window_end_idx)
            ])

            # Motif indexes for X_indices (only motifs fully starting within the window)
            motif_in_lookback = sorted([
                int(motif_idx) - idx
                for motif_idx in motif_indexes
                if idx <= motif_idx < window_end_idx
            ])
            # Motif indexes in the forecast period
            motif_in_forecast = sorted([
                int(motif_idx)
                for motif_idx in motif_indexes
                if window_end_idx <= motif_idx < forecast_period_end
            ])
             # if motif has index in the lookback window and forecast period
            if len(motif_in_lookback) >= 2 and len(motif_in_forecast) >= 1 :
                valid_instance = True

                # Compute distance to the nearest motif  in the forecast period
                motif_indexes_in_window.append(motif_in_lookback)
                forecast_distances.append(min(motif_in_forecast) - window_end_idx + 1)

                # Update the mask for the motifs in the lookback window
                for motif_start in motif_in_mask:
                    motif_end = motif_start + motif_size
                    if motif_start < lookback_period and motif_end > 0:
                        mask_window[max(0, motif_start):min(lookback_period, motif_end)] = 1
                mask_windows.append(mask_window)
                    
            else:
                continue  # ignore motifs that are not in the lookback window and forecast period
        
        if not valid_instance:
            continue  # Skip instances without any motifs in the forecast period

        # Append to the dataset
        for i in range(len(motif_indexes_in_window)):
            X1.append(torch.tensor(data_window, dtype=torch.float32))
            X2.append(motif_indexes_in_window[i])
            y.append(torch.tensor(forecast_distances[i], dtype=torch.float32))
            mask.append(mask_windows[i])

    # Stack the results
    X1 = torch.stack(X1)  # Shape: (num_samples, lookback_period, num_features)
    X2_padded = pad_sequence([torch.tensor(motif_indexes, dtype=torch.float32) for motif_indexes in X2], batch_first=True, padding_value=-1).unsqueeze(-1)  # Shape: (num_samples, max_num_motifs, max_num_repetitions)
    y = torch.stack(y).unsqueeze(1)  # Shape: (num_samples,1)
    mask = torch.stack(mask)  # Shape: (num_samples, lookback_period)

    return X1, X2_padded, mask, y



def print_study_results(study):
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:", study.best_trial.number)
    print("Best hyperparameters:", study.best_params)
    print("Validation Losses:", [round(loss, 3) for loss in study.best_trial.user_attrs["fold_val_losses"]])
    print("Mean validation loss:", round(study.best_trial.user_attrs["mean_val_loss"], 3))
    print("Test Losses:", [round(loss, 3) for loss in study.best_trial.user_attrs["test_losses"]])
    print("Mean test loss:", round(study.best_trial.user_attrs["mean_test_loss"], 3))
    #print("Test MAE:", [round(mae, 3) for mae in study.best_trial.user_attrs["test_mae_per_fold"]])
    print("Mean test MAE:", round(study.best_trial.user_attrs["mean_test_mae"], 3), 
          "std:", round(study.best_trial.user_attrs["std_test_mae"], 3))
    #print("Test RMSE:", [round(rmse, 3) for rmse in study.best_trial.user_attrs["test_rmse_per_fold"]])
    print("Mean test RMSE:", round(study.best_trial.user_attrs["mean_test_rmse"], 3), 
          "std:", round(study.best_trial.user_attrs["std_test_rmse"], 3))

def get_best_model_results_traindevtest(study):
    best_trial = study.best_trial
    train_losses = best_trial.user_attrs["train_losses"]
    val_losses = best_trial.user_attrs["validation_losses"]
    best_epoch = best_trial.user_attrs["best_epoch"]
    test_loss = best_trial.user_attrs["test_loss"]
    test_mae = best_trial.user_attrs["test_mae"]
    test_rmse = best_trial.user_attrs["test_rmse"]
    return train_losses, val_losses, best_epoch, test_loss, test_mae, test_rmse

def get_best_model_results(study):
    best_trial = study.best_trial
    fold_val_losses = best_trial.user_attrs["fold_val_losses"]
    fold_test_losses = best_trial.user_attrs["test_losses"]
    return fold_val_losses, fold_test_losses


def plot_best_model_results(study_df, save_path=None):
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
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_best_model_results_traindevtest(study_df, save_path=None):

    train_losses = study_df["user_attrs_train_losses"].iloc[study_df["value"].idxmin()]
    val_losses = study_df["user_attrs_validation_losses"].iloc[study_df["value"].idxmin()]
    
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()




def plot_preds_vs_truevalues(true_values, predictions, fold, save_path=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true_values, mode='markers', name='True Values'))
    fig.add_trace(go.Scatter(y=predictions, mode='markers', name='Predictions'))
    fig.update_layout(
        title=f"Fold {fold} - True Values vs Predictions",
        xaxis_title="Sample",
        yaxis_title="Value"
    )
    if save_path:
        fig.write_image(save_path)
    fig.show()
