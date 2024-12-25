import torch
from torch import nn

class BaselineAverageModel(nn.Module):
    def __init__(self):
        """
        Initialize the baseline model.

        """
        super(BaselineAverageModel, self).__init__()

    def forward(self, X=None, mask=None, indexes=None):
        """
        Forward pass: calculate the average of the differences between consecutive values.

        Args:
        - indexes (Tensor): Input tensor, size (batch_size, window_len, 1).

        Returns:
        - Tensor: Average differences, size (batch_size, 1).
        """
        # Ensure indexes is float and maintain its device
        indexes = indexes.float()
        batch_avg_differences = []
        device = indexes.device  # Get the device of the input tensor

        for batch in indexes:  # Iterate over batches
            valid_values = batch[batch != -1]  # Remove padding (-1 values)
            if len(valid_values) <= 1:  # Not enough values to compute differences
                batch_avg_differences.append(torch.tensor(0.0, device=device))
            else:
                differences = valid_values[1:] - valid_values[:-1]  # Consecutive differences
                avg_difference = torch.mean(differences)
                batch_avg_differences.append(avg_difference)

        return torch.tensor(batch_avg_differences, device=device).unsqueeze(1)  # Return as (batch_size, 1)



        

class Baseline_NaiveLastDifference(nn.Module):
    def __init__(self):
        """
        Initialize the baseline model.

        """
        super(Baseline_NaiveLastDifference, self).__init__()

    def forward(self, X=None, mask = None, indexes=None):
        """
        Forward pass: return the last value in the sequence.

        Args:
        - indexes (Tensor): Input tensor, size (batch_size, window_len, 1).

        Returns:
        - Tensor: difference between the last two values in the sequence, size (batch_size, 1).
        """
        # Remove padding in window_len, preserving batch structure
        indexes = indexes.float()
        batch_results = []

        for batch in indexes:  # Iterate over batches
            valid_values = batch[batch != -1]
            if len(valid_values) == 0:
                batch_results.append(torch.tensor(0.0, device=indexes.device))
            else:
                if len(valid_values) == 1:
                    batch_results.append(valid_values[0])
                else:
                    batch_results.append(valid_values[-1] - valid_values[-2])

        return torch.tensor(batch_results, device=indexes.device).unsqueeze(1)  # Return as (batch_size, 1)
    
