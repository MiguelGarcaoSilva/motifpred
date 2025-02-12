import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, sequence_length,
                 num_filters_list, kernel_size, pool_size, output_dim):
        """
        Unified CNN model with optional mask input.

        Args:
            input_channels (int): Number of input channels.
            sequence_length (int): Length of the input sequence.
            output_dim (int): Dimension of the output.
            num_filters_list (list of int): Number of filters for each convolutional layer.
            kernel_size (int): Kernel size for each convolutional layer.
            pool_size (int or None): Size of the pooling window for max pooling. If None, pooling is skipped.
            hidden_units (int): Number of hidden units in the fully connected layer.
        """
        super(CNN, self).__init__()
        

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
                    kernel_size=kernel_size, 
                    stride=1, 
                    padding=kernel_size// 2  # To maintain input size
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

        # Fully connected layer
        self.fc = nn.Linear(num_filters_list[-1] * current_sequence_length, output_dim)

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_channels).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """

        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)

        # Pass through convolutional and pooling layers
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pooling_layers[i](x)

        # Flatten for the fully connected layer
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, -1)

        # Pass through fully connected layer
        x = self.fc(x)
        return x
