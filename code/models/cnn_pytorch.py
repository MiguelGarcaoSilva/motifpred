import torch
from torch import nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNX1(nn.Module):
    def __init__(self, input_channels, sequence_length, output_dim, 
                 num_filters_list, kernel_sizes_list, pool_size=None):
        """
        Generalized CNN with variable number of convolutional layers, filter sizes, and optional pooling.

        Args:
            input_channels (int): Number of input channels.
            sequence_length (int): Length of the input sequence.
            output_dim (int): Dimension of the output.
            num_filters_list (list of int): Number of filters for each convolutional layer.
            kernel_sizes_list (list of int): Kernel size for each convolutional layer.
            pool_size (int or None): Size of the pooling window for max pooling. If None, pooling is skipped.
        """
        super(CNNX1, self).__init__()
        
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

        self.fc = nn.Linear(num_filters_list[-1] * current_sequence_length, output_dim)

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)

        # Pass through convolutional and pooling layers
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pooling_layers[i](x)

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)

        # Pass through fully connected layers
        x = self.fc(x)  # Direct output

        return x


class CNNX1_X2Masking(nn.Module):
    def __init__(self, input_channels, sequence_length, output_dim, 
                 num_filters_list, kernel_sizes_list, pool_size=None):
        """
        Generalized CNN with variable number of convolutional layers, filter sizes, and optional pooling.

        Args:
            input_channels (int): Number of input channels.
            sequence_length (int): Length of the input sequence.
            output_dim (int): Dimension of the output.
            num_filters_list (list of int): Number of filters for each convolutional layer.
            kernel_sizes_list (list of int): Kernel size for each convolutional layer.
            pool_size (int or None): Size of the pooling window for max pooling. If None, pooling is skipped.
        """
        super(CNNX1_X2Masking, self).__init__()

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

        self.fc = nn.Linear(num_filters_list[-1] * current_sequence_length, output_dim)

    def forward(self, x, mask):
        # Input x shape: (batch_size, sequence_length, features)
        # mask is size (batch_size, sequence_length)
        mask = mask.unsqueeze(-1)  # mask is size (batch_size, sequence_length, 1)
        x = torch.cat((x, mask), dim=2)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)

        # Pass through convolutional and pooling layers
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if self.use_pooling:
                x = self.pooling_layers[i](x)

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)

        # Pass through fully connected layers
        x = self.fc(x)  # Direct output

        return x



#TODO: implement. how?
#class CNNX1_X2Indices(nn.Module):
