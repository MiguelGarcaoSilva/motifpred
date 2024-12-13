import torch
from pytorch_tcn import TCN

class TCNX1(torch.nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout):
        super(TCNX1, self).__init__()
        # Define TCN
        self.tcn = TCN(
            num_inputs=input_channels,  # Number of input variables
            num_channels=num_channels,  # Filters for residual blocks
            kernel_size=kernel_size,  # Size of convolution kernel
            dropout=dropout,  # Dropout rate
            causal=False,  
            input_shape="NLC",  # Input shape convention
            output_projection=None,  # Do not directly project
        )
        # Fully connected layer to map features to single output
        self.fc = torch.nn.Linear(num_channels[-1], 1)  # Map to scalar value

    def forward(self, x):
        """
        Forward pass.

        Parameters:
        - x: Input tensor of shape (batch_size, sequence_length, input_channels).

        Returns:
        - Tensor: Predicted value of shape (batch_size, 1).
        """
        # Pass through TCN
        tcn_output = self.tcn(x)  # Shape: (batch_size, sequence_length, num_channels[-1])
        # Use last time step's features
        last_step_features = tcn_output[:, -1, :]  # Shape: (batch_size, num_channels[-1])
        # Predict scalar output
        output = self.fc(last_step_features)  # Shape: (batch_size, 1)
        return output

class TCNSeries_X2Masking(torch.nn.Module):

    def __init__(self, input_channels, num_channels, kernel_size, dropout):
        super(TCNSeries_X2Masking, self).__init__()
        self.tcn = TCN(
            num_inputs=input_channels,  # Number of input variables
            num_channels=num_channels,  # Filters for residual blocks
            kernel_size=kernel_size,  # Size of convolution kernel
            dropout=dropout,  # Dropout rate
            causal=False,  # Use causal convolutions
            input_shape="NLC",  # Input shape convention
            output_projection=None,  # Do not directly project
        )
        self.fc = torch.nn.Linear(num_channels[-1], 1)

    def forward(self, x, mask):
        # X1 is size (batch_size, windown_len, features)
        # mask is size (batch_size, windown_len)
        mask = mask.unsqueeze(-1)
        x = torch.cat((x, mask), dim=2)
        tcn_output = self.tcn(x)
        last_step_features = tcn_output[:, -1, :]
        output = self.fc(last_step_features)

        return output