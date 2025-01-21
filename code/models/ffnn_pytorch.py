import torch
from torch import nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim=1):
        super(FFNN, self).__init__()

        # Define layers
        self.input_layer = nn.Linear(input_dim, hidden_sizes_list[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]) for i in range(len(hidden_sizes_list) - 1)])
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
        - x (Tensor): Main input tensor, size (batch_size, window_len * features).
        Returns:
        - Tensor: Output of the network.
        """
        # Pass through the network
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
