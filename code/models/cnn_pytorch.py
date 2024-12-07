import torch
from torch import nn
import torch.nn.functional as F


class CNNX1(nn.Module):
    def __init__(self, input_channels, sequence_length, output_dim, 
                 num_filters_1, num_filters_2, kernel_size):
        super(CNNX1, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=num_filters_1, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2  # To maintain input size
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters_1, 
            out_channels=num_filters_2, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2
        )
        
        # Calculate the output size after two convolutional layers
        conv_output_size = sequence_length  # No pooling, so sequence length remains unchanged
        
        self.fc1 = nn.Linear(num_filters_2 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_dim)

    
    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, channels, sequence_length)

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))  # Apply first convolution
        x = F.relu(self.conv2(x))  # Apply second convolution

        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, -1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)          # Regression output

        return x



class CNNX1_X2Masking(nn.Module):
    def __init__(self, input_channels, sequence_length, output_dim, 
                 num_filters_1, num_filters_2, kernel_size):
        super(CNNX1_X2Masking, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=num_filters_1, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2  # To maintain input size
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters_1, 
            out_channels=num_filters_2, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2
        )
        
        # Calculate the output size after two convolutional layers
        conv_output_size = sequence_length

        self.fc1 = nn.Linear(num_filters_2 * conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_dim)


    def forward(self, x, mask):
        # X1 is size (batch_size, windown_len, features)
        # mask is size (batch_size, windown_len)
        mask = mask.unsqueeze(-1)  #mask is size (batch_size, windown_len, 1)
        x = torch.cat((x, mask), dim=2)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, channels, sequence_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#TODO: implement. how?
#class CNNX1_X2Indices(nn.Module):
