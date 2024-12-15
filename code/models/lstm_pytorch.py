import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim, auxiliary_input_dim=0):
        """
        Unified LSTM model that can optionally process auxiliary inputs (e.g., masks).

        Args:
            input_dim (int): Dimension of the primary input features.
            hidden_sizes_list (list of int): List specifying hidden size for each LSTM layer.
            output_dim (int): Dimension of the output features.
            auxiliary_input_dim (int): Dimension of the auxiliary input features. Default is 0.
        """
        super(LSTM, self).__init__()
        self.num_layers = len(hidden_sizes_list)

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        in_dim = input_dim + auxiliary_input_dim  # Adjust input size based on auxiliary input
        for hidden_size in hidden_sizes_list:
            self.lstm_layers.append(nn.LSTM(input_size=in_dim, hidden_size=hidden_size, batch_first=True))
            in_dim = hidden_size

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, primary_input, auxiliary_input=None):
        """
        Forward pass through the LSTM.

        Args:
            primary_input (torch.Tensor): Primary input tensor of shape (batch_size, seq_len, input_dim).
            auxiliary_input (torch.Tensor, optional): Auxiliary input tensor of shape (batch_size, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if auxiliary_input is not None:
            # Concatenate auxiliary input along the feature dimension
            auxiliary_input = auxiliary_input.unsqueeze(-1)  # Add feature dimension to mask
            x = torch.cat((primary_input, auxiliary_input), dim=2)
        else:
            x = primary_input

        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        # Use the last hidden state for output
        x = x[:, -1, :]  # Select the last time step's output
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


