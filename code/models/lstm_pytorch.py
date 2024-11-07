import torch
from torch import nn

# Model 1: LSTM processing only primary input (X1)
class LSTMX1Input(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMX1Input, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing primary input (X1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, primary_input):
        # Forward propagate through LSTM with primary input only
        _, (hidden_state, _) = self.lstm(primary_input)

        # Pass through the output layer
        return self.output_layer(hidden_state[0])

# Model 2: LSTM with secondary input added after LSTM layer
class LSTMX1_X2AfterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, auxiliary_input_dim):
        super(LSTMX1_X2AfterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing primary input (X1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer that takes concatenated LSTM output and auxiliary input
        self.output_layer = nn.Linear(hidden_size + auxiliary_input_dim, output_size)
        
    def forward(self, primary_input, auxiliary_input):
        # Forward propagate through LSTM with primary input
        _, (hidden_state, _) = self.lstm(primary_input)

        # Concatenate hidden state output with auxiliary input along the feature dimension
        combined_input = torch.cat((hidden_state[0], auxiliary_input.float()), dim=1)

        # Pass through the output layer
        return self.output_layer(combined_input)

# Model 3: LSTM with secondary input concatenated before the LSTM layer
class LSTMX1_X2BeforeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, auxiliary_input_dim):
        super(LSTMX1_X2BeforeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer that processes concatenated primary and auxiliary input
        self.lstm = nn.LSTM(input_size + auxiliary_input_dim, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, primary_input, auxiliary_input):
        # Expand and repeat auxiliary input to match sequence length of primary input
        auxiliary_expanded = auxiliary_input.unsqueeze(1).repeat(1, primary_input.shape[1], 1)
        concatenated_input = torch.cat((primary_input, auxiliary_expanded), dim=2)

        # Forward propagate through LSTM with concatenated input
        _, (hidden_state, _) = self.lstm(concatenated_input)

        # Pass through the output layer
        return self.output_layer(hidden_state[0])
