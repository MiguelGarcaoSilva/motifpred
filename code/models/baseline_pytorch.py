import torch
from torch import nn

class BaselineAverage(nn.Module):
    def __init__(self, n_timepoints):
        """
        Initialize the baseline model.
        """
        super(BaselineAverage, self).__init__()
        self.n_timepoints = n_timepoints

    def forward(self, indexes):
        """
        Forward pass: predict the timepoints needed for a new repetition after X ends.

        Args:
        - X (Tensor): Input tensor, size (batch_size, window_len, feature_dim).
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

                last_index = valid_values[-1]
                next_prediction = last_index + avg_difference
                time_to_next_repetition = next_prediction - self.n_timepoints
                # default when time_to_next_repetition is before end of lookpack period
                time_to_next_repetition = 1 if time_to_next_repetition < 1 else time_to_next_repetition
                batch_predictions.append(time_to_next_repetition)

        return torch.tensor(batch_predictions, device=device).unsqueeze(1)  # Return as (batch_size, 1)





        

class BaselineLastDifference(nn.Module):
    def __init__(self, n_timepoints):
        """
        Initialize the baseline model.

        """
        super(BaselineLastDifference, self).__init__()
        self.n_timepoints = n_timepoints


    def forward(self, indexes):
        """
        Forward pass: predict the timepoints needed for a new repetition after X ends.

        Args:
        - X (Tensor): Input tensor, size (batch_size, window_len, feature_dim).
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

                last_index = valid_values[-1]  # Last index in the sequence
                next_prediction = last_index + last_difference  # Next index prediction
                # Ensure the next prediction is after the end of the window
                time_to_next_repetition = next_prediction - self.n_timepoints 
                time_to_next_repetition = 1 if time_to_next_repetition < 1 else time_to_next_repetition
                batch_predictions.append(time_to_next_repetition)

        return torch.tensor(batch_predictions, device=indexes.device).unsqueeze(1)  # Return as (batch_size, 1)
