from matplotlib import pyplot as plt
import numpy as np
from utils.timeseries_split import BlockingTimeSeriesSplit
from torch.nn.utils.rnn import pad_sequence
import torch
import plotly.graph_objects as go

def create_dataset(data, lookback_period, step, forecast_period, motif_indexes):
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
        data_window = data[:, idx:window_end_idx].T

        # Calculate `y`
        data_y = -1
        if next_match_in_forecast_period != -1:
            # Index of the next match relative to the end of the window
            data_y = next_match_in_forecast_period - window_end_idx
        
        # Append to lists
        X1.append(torch.tensor(data_window, dtype=torch.float32))  # Now with shape (lookback_period, num_features)
        X2.append(torch.tensor(motif_indexes_in_window, dtype=torch.float32)) 
        y.append(data_y) 

    # Pad X2 sequences to have the same length
    X2_padded = pad_sequence(X2, batch_first=True, padding_value=-1).unsqueeze(-1) # Final shape: (num_samples, max_num_motifs, 1)

    # Convert lists to torch tensors
    X1 = torch.stack(X1)  # Final shape: (num_samples, lookback_period, num_features)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    return X1, X2_padded, y

def print_study_results(study):
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:", study.best_trial.number)
    print("Best hyperparameters:", study.best_params)
    print("Validation Losses:", [round(loss, 3) for loss in study.best_trial.user_attrs["fold_val_losses"]])
    print("Mean validation loss:", round(study.best_trial.user_attrs["mean_val_loss"], 3))
    print("Test Losses:", [round(loss, 3) for loss in study.best_trial.user_attrs["test_losses"]])
    print("Mean test loss:", round(study.best_trial.user_attrs["mean_test_loss"], 3))
    print("Mean test MAE:", round(study.best_trial.user_attrs["mean_test_mae"], 3), 
          "std:", round(study.best_trial.user_attrs["std_test_mae"], 3))
    print("Mean test RMSE:", round(study.best_trial.user_attrs["mean_test_rmse"], 3), 
          "std:", round(study.best_trial.user_attrs["std_test_rmse"], 3))


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