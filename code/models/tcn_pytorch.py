import torch
from pytorch_tcn import TCN

class MYTCN(torch.nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, dropout):
        """
        Unified TCN model with optional masking input.

        Args:
            input_channels (int): Number of input channels.
            num_channels (list of int): Number of filters in each residual block.
            kernel_size (int): Kernel size for convolutional layers.
            dropout (float): Dropout rate.
        """
        super(MYTCN, self).__init__()
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

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_channels).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, sequence_length). Default is None.

        Returns:
            torch.Tensor: Predicted value of shape (batch_size, 1).
        """
        if mask is not None:
            # Add a feature dimension to the mask and concatenate with x
            mask = mask.unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
            x = torch.cat((x, mask), dim=2)

        # Pass through TCN
        tcn_output = self.tcn(x)  # Shape: (batch_size, sequence_length, num_channels[-1])
        # Use last time step's features
        last_step_features = tcn_output[:, -1, :]  # Shape: (batch_size, num_channels[-1])
        # Predict scalar output
        output = self.fc(last_step_features)  # Shape: (batch_size, 1)
        return output
