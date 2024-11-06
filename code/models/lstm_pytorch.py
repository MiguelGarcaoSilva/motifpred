from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing X1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, X1):
        
        # Forward propagate LSTM
        _, (h_n, _) = self.lstm(X1)

        # Pass through the final fully connected layer
        out = self.fc(h_n[0])
        return out

#TODO: check if this is how i want to deal with the x2 data
class LSTMModel_x2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for processing X1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc = nn.Linear(hidden_size + X2.shape[1], output_size)
        
    def forward(self, X1, X2):
        
        # Forward propagate LSTM
        _, (h_n, _) = self.lstm(X1)

        # Concatenate with X2 (motif indexes)
        out = torch.cat((h_n[0], X2.float()), dim=1)  # Concatenate along the feature dimension

        # Pass through the final fully connected layer
        out = self.fc(out)
        return out