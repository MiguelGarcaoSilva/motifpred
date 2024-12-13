import torch
from torch import nn
import torch.nn.functional as F

# Model 1: LSTM processing only primary input (X1)
class LSTMX1(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim):
        """
        Generalized LSTM with variable hidden sizes for each layer.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_sizes_list (list of int): List specifying hidden size for each LSTM layer.
            output_dim (int): Dimension of the output features.
        """
        super(LSTMX1, self).__init__()
        self.num_layers = len(hidden_sizes_list)

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        in_dim = input_dim
        for hidden_size in hidden_sizes_list:
            self.lstm_layers.append(nn.LSTM(input_size=in_dim, hidden_size=hidden_size, batch_first=True))
            in_dim = hidden_size

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, x):
        """
        Forward pass through the LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for lstm in self.lstm_layers:
            x, _ = lstm(x)  # Forward through each LSTM layer

        # Use the last hidden state for output
        x = x[:, -1, :]  # Select the last time step's output
        return self.output_layer(x)


# Model: LSTM with X1 time series and X2 is masking of indices added as extra feature
class LSTMX1Series_X2Masking(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim, auxiliary_input_dim):
        """
        Args:
            input_dim (int): Dimension of the primary input features.
            hidden_sizes_list (list of int): List specifying hidden size for each LSTM layer.
            output_dim (int): Dimension of the output features.
            auxiliary_input_dim (int): Dimension of the auxiliary input features.
        """
        super(LSTMX1Series_X2Masking, self).__init__()
        self.num_layers = len(hidden_sizes_list)

        # LSTM layers for processing primary input (X1 + mask)
        self.lstm_layers = nn.ModuleList()
        in_dim = input_dim + auxiliary_input_dim
        for hidden_size in hidden_sizes_list:
            self.lstm_layers.append(nn.LSTM(input_size=in_dim, hidden_size=hidden_size, batch_first=True))
            in_dim = hidden_size

        # Fully connected layer that takes LSTM output
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, primary_input, mask_input):
        """
        Forward pass through the LSTM with primary input and mask.

        Args:
            primary_input (torch.Tensor): Primary input tensor of shape (batch_size, seq_len, input_dim).
            mask_input (torch.Tensor): Mask input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        concatenated_input = torch.cat((primary_input, mask_input.unsqueeze(-1)), dim=2)

        for lstm in self.lstm_layers:
            concatenated_input, _ = lstm(concatenated_input)

        x = concatenated_input[:, -1, :]  # Select the last time step's output
        return self.output_layer(x)



# Model: LSTM with secondary input concatenated before the LSTM layer
class LSTMX1Series_X2Indices(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim, auxiliary_input_dim):
        """
        Args:
            input_dim (int): Dimension of the primary input features.
            hidden_sizes_list (list of int): List specifying hidden size for each LSTM layer.
            output_dim (int): Dimension of the output features.
            auxiliary_input_dim (int): Dimension of the auxiliary input features.
        """
        super(LSTMX1Series_X2Indices, self).__init__()
        self.num_layers = len(hidden_sizes_list)

        # LSTM layers for processing concatenated input
        self.lstm_layers = nn.ModuleList()
        in_dim = input_dim + auxiliary_input_dim
        for hidden_size in hidden_sizes_list:
            self.lstm_layers.append(nn.LSTM(input_size=in_dim, hidden_size=hidden_size, batch_first=True))
            in_dim = hidden_size

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, primary_input, auxiliary_input):
        """
        Forward pass through the LSTM with concatenated primary and auxiliary input.

        Args:
            primary_input (torch.Tensor): Primary input tensor of shape (batch_size, seq_len, input_dim).
            auxiliary_input (torch.Tensor): Auxiliary input tensor of shape (batch_size, auxiliary_input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        auxiliary_expanded = auxiliary_input.unsqueeze(1).repeat(1, primary_input.shape[1], 1)
        concatenated_input = torch.cat((primary_input, auxiliary_expanded), dim=2)

        for lstm in self.lstm_layers:
            concatenated_input, _ = lstm(concatenated_input)

        x = concatenated_input[:, -1, :]  # Select the last time step's output
        return self.output_layer(x)


