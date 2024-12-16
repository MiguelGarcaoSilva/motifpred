import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class TCNModel(torch.nn.Module):
    def __init__(self, input_channels, num_channels_list, kernel_size, dropout):
        """
        Unified TCN model with optional masking input.

        Args:
            input_channels (int): Number of input channels.
            num_channels_list (list of int): Number of filters in each residual block.
            kernel_size (int): Kernel size for convolutional layers.
            dropout (float): Dropout rate.
        """
        super(TCNModel, self).__init__()
        # Define TCN
        self.tcn = TCN(
            num_inputs=input_channels,  # Number of input variables
            num_channels=num_channels_list,  # Filters for residual blocks
            kernel_size=kernel_size,  # Size of convolution kernel
            dropout=dropout,  # Dropout rate
            causal=False,  
            input_shape="NLC",  # Input shape convention
            output_projection=None,  # Do not directly project
        )
        # Fully connected layer to map features to single output
        self.fc = torch.nn.Linear(num_channels_list[-1], 1)  # Map to scalar value

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



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, num_channels_list, kernel_size=2, dropout=0.2, output_dim = 1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels_list)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels_list[i-1]
            out_channels = num_channels_list[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_dim)


    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)
        y1 = self.network(x)
        o = self.linear(y1[:, :, -1]) # just take the last time step features
        return o 
