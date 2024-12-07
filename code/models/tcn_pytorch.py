import torch
from pytorch_tcn import TCN

class TCNModel(torch.nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout, output_dim, output_activation, causal, pooling_type="mean"):
        super(TCNModel, self).__init__()
        # Define the TCN model
        self.tcn = TCN(
            num_inputs=input_channels,  # Number of input features (time series variables)
            num_channels=num_channels,  # Filters in each residual block
            kernel_size=kernel_size,  # Size of convolutional kernel
            dropout=dropout,  # Dropout rate
            output_projection=output_dim,  # Project to desired output dimension
            output_activation=None,  # No activation for regression
            causal=True,  # Causal convolutions
            input_shape="NLC"  # Input shape: (batch_size, sequence_length, num_features)
        )
        # Pooling type to summarize sequence information
        self.pooling_type = pooling_type

    def forward(self, x):
        """
        Forward pass through the TCN.
        
        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_channels).

        Returns:
        - Tensor: Final output of shape (batch_size, output_dim).
        """
        tcn_output = self.tcn(x)  # Shape: (batch_size, sequence_length, output_dim)
        
        # Summarize the sequence dimension
        if self.pooling_type == "mean":
            # Mean pooling
            pooled_output = torch.mean(tcn_output, dim=1)  # Shape: (batch_size, output_dim)
        elif self.pooling_type == "max":
            # Max pooling
            pooled_output, _ = torch.max(tcn_output, dim=1)  # Shape: (batch_size, output_dim)
        elif self.pooling_type == "last":
            # Use the last time step's output
            pooled_output = tcn_output[:, -1, :]  # Shape: (batch_size, output_dim)
        else:
            raise ValueError("Invalid pooling type. Choose from 'mean', 'max', or 'last'.")
        
        return pooled_output
