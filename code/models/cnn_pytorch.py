import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, sequence_length, output_dim, 
                 num_filters_list, kernel_sizes_list, pool_size=None, hidden_units=128):
        """
        Unified CNN model with optional mask input.

        Args:
            input_channels (int): Number of input channels.
            sequence_length (int): Length of the input sequence.
            output_dim (int): Dimension of the output.
            num_filters_list (list of int): Number of filters for each convolutional layer.
            kernel_sizes_list (list of int): Kernel size for each convolutional layer.
            pool_size (int or None): Size of the pooling window for max pooling. If None, pooling is skipped.
            hidden_units (int): Number of hidden units in the fully connected layer.
        """
        super(CNN, self).__init__()
        
        assert len(num_filters_list) == len(kernel_sizes_list), (
            "Number of convolutional layers must match the lengths of num_filters_list and kernel_sizes_list."
        )

        self.use_pooling = pool_size is not None
        num_conv_layers = len(num_filters_list)

        self.conv_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList() if self.use_pooling else None
        in_channels = input_channels

        current_sequence_length = sequence_length
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels, 
                    out_channels=num_filters_list[i], 
                    kernel_size=kernel_sizes_list[i], 
                    stride=1, 
                    padding=kernel_sizes_list[i] // 2  # To maintain input size
                )
            )
            if self.use_pooling:
                if current_sequence_length < pool_size:
                    raise ValueError(f"Pooling size {pool_size} too large for sequence length {current_sequence_length} at layer {i}.")
                self.pooling_layers.append(
                    nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
                )
                current_sequence_length //= pool_size
            in_channels = num_filters_list[i]

        if current_sequence_length <= 0:
            raise ValueError(f"Invalid sequence length after convolution and pooling: {current_sequence_length}. Reduce the number of layers or pooling size.")

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters_list[-1] * current_sequence_length, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_channels).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, sequence_length). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if mask is not None:
            # Add a feature dimension to the mask and concatenate it to the input
            mask = mask.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
            x = torch.cat((x, mask), dim=2)

        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)

        # Pass through convolutional and pooling layers
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pooling_layers[i](x)

        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x
