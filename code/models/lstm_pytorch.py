import torch
from torch import nn
import torch.nn.functional as F

# Model 1: LSTM processing only primary input (X1)
class LSTMX1(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim):
        super(LSTMX1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing primary input (X1)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        # Forward propagate through LSTM with primary input only
        _, (hidden_state, _) = self.lstm(x)

        # Pass through the output layer
        return self.output_layer(hidden_state[0])


# Model: LSTM with attention mechanism
class LSTMAttentionX1(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim):
        super(LSTMAttentionX1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing primary input (X1)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)

        # Attention layer: a linear layer that produces the attention scores
        self.attention_layer = nn.Linear(hidden_size, 1, bias=False)

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        # Forward propagate through LSTM with primary input only
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        
        # Apply attention
        # Compute attention scores for each time step
        attention_scores = self.attention_layer(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)         # Normalize scores with softmax

        # Compute the context vector as a weighted sum of lstm_out (hidden states)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)

        # Pass the context vector through the output layer
        output = self.output_layer(context_vector)
        return output


# Model: LSTM with X1 time series and X2 is masking of indices added as extra feature
class LSTMX1_X2Masking(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, auxiliary_input_dim):
        super(LSTMX1_X2Masking, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing primary input (X1 + mask)
        #input dim is the number of features
        self.lstm = nn.LSTM(input_dim + auxiliary_input_dim, hidden_size, num_layers, batch_first=True)

        # Fully connected layer that takes concatenated LSTM output and mask input
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
    def forward(self, primary_input, mask_input):
        
        # X1 is size (batch_size, seq_len, input_dim)
        # mask_input is size (batch_size, seq_len)
        concatenated_input = torch.cat((primary_input, mask_input.unsqueeze(-1)), dim=2)

        # Forward propagate through LSTM with primary input
        _, (hidden_state, _) = self.lstm(concatenated_input)

        # Pass through the output layer
        return self.output_layer(hidden_state[0])


# Model: LSTM with secondary input concatenated before the LSTM layer
class LSTMX1_X2BeforeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, auxiliary_input_dim):
        super(LSTMX1_X2BeforeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer that processes concatenated primary and auxiliary input
        self.lstm = nn.LSTM(input_dim + auxiliary_input_dim, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to produce the output
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
    def forward(self, primary_input, auxiliary_input):
        # Expand and repeat auxiliary input to match sequence length of primary input
        auxiliary_expanded = auxiliary_input.unsqueeze(1).repeat(1, primary_input.shape[1], 1)
        concatenated_input = torch.cat((primary_input, auxiliary_expanded), dim=2)

        # Forward propagate through LSTM with concatenated input
        _, (hidden_state, _) = self.lstm(concatenated_input)

        # Pass through the output layer
        return self.output_layer(hidden_state[0])

# Model: LSTM with secondary input added after LSTM layer
class LSTMX1_X2AfterLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, auxiliary_input_dim):
        super(LSTMX1_X2AfterLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing primary input (X1)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)

        # Fully connected layer that takes concatenated LSTM output and auxiliary input
        self.output_layer = nn.Linear(hidden_size + auxiliary_input_dim, output_dim)
        
    def forward(self, primary_input, auxiliary_input):
        # Forward propagate through LSTM with primary input
        _, (hidden_state, _) = self.lstm(primary_input)

        # Concatenate hidden state output with auxiliary input along the feature dimension
        combined_input = torch.cat((hidden_state[0], auxiliary_input.float()), dim=1)

        # Pass through the output layer
        return self.output_layer(combined_input)









        
