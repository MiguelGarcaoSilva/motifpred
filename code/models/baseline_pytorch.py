import torch
from torch import nn

class BaselineAverageModel(nn.Module):
    def __init__(self):
        """
        Initialize the baseline model.
        """
        super(BaselineAverageModel, self).__init__()

    def forward(self, X, mask, indexes):
        """
        Forward pass: predict the timepoints needed for a new repetition after X ends.

        Args:
        - X (Tensor): Input tensor, size (batch_size, window_len, feature_dim).
        - mask (Tensor): Optional mask tensor, size (batch_size, window_len).
        - indexes (Tensor): Input tensor, size (batch_size, window_len, 1).

        Returns:
        - Tensor: Predicted timepoints, size (batch_size, 1).
        """
        # Ensure indexes is float and maintain its device
        indexes = indexes.float()
        batch_predictions = []
        device = indexes.device  # Get the device of the input tensor

        for batch in indexes:  # Iterate over batches
            valid_values = batch[batch != -1]  # Remove padding (-1 values)
            if len(valid_values) <= 1:  # Not enough values to compute differences
                batch_predictions.append(torch.tensor(0.0, device=device))
            else:
                differences = valid_values[1:] - valid_values[:-1]  # Consecutive differences
                avg_difference = torch.mean(differences)  # Average interval


                x_length = X.size(1)  # Length of X (timepoints)

                last_index = valid_values[-1]
                next_prediction = last_index + avg_difference
                time_to_next_repetition = next_prediction - x_length
                time_to_next_repetition = 0 if time_to_next_repetition < 0 else time_to_next_repetition
                batch_predictions.append(time_to_next_repetition)

        return torch.tensor(batch_predictions, device=device).unsqueeze(1)  # Return as (batch_size, 1)





        

class Baseline_NaiveLastDifference(nn.Module):
    def __init__(self):
        """
        Initialize the baseline model.

        """
        super(Baseline_NaiveLastDifference, self).__init__()


    def forward(self, X, mask, indexes):
        """
        Forward pass: predict the timepoints needed for a new repetition after X ends.

        Args:
        - X (Tensor): Input tensor, size (batch_size, window_len, feature_dim).
        - mask (Tensor): Optional mask tensor, size (batch_size, window_len).
        - indexes (Tensor): Input tensor, size (batch_size, window_len, 1).

        Returns:
        - Tensor: Predicted timepoints, size (batch_size, 1).
        """
        # Remove padding in window_len, preserving batch structure
        indexes = indexes.float()
        batch_predictions = []

        for batch in indexes:  # Iterate over batches
            valid_values = batch[batch != -1]  # Remove padding (-1 values)
            if len(valid_values) <= 1:
                batch_predictions.append(torch.tensor(0.0, device=indexes.device))
            else:
                # Use the last difference as the interval
                last_difference = valid_values[-1] - valid_values[-2]
                x_length = X.size(1)  # Length of X (timepoints)

                last_index = valid_values[-1]  # Last index in the sequence
                next_prediction = last_index + last_difference  # Next index prediction
                time_to_next_repetition = next_prediction - x_length 
                time_to_next_repetition = 0 if time_to_next_repetition < 0 else time_to_next_repetition
                batch_predictions.append(time_to_next_repetition)

        return torch.tensor(batch_predictions, device=indexes.device).unsqueeze(1)  # Return as (batch_size, 1)
