import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout=0.1,
        seq_length=10
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(seq_length, d_model)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Regression output layer
        self.regression_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)

        # Embedding
        x = self.input_embedding(x)  # Shape: (batch_size, seq_length, d_model)
        x += self.positional_encoding[:x.size(1), :]  # Add positional encoding

        # Reshape for transformer: (seq_length, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # Transformer encoder
        x = self.transformer_encoder(x)  # Shape: (seq_length, batch_size, d_model)

        # Take the representation of the last time step
        x = x[-1, :, :]  # Shape: (batch_size, d_model)

        # Regression output
        output = self.regression_head(x)  # Shape: (batch_size, 1)

        return output

    def _generate_positional_encoding(self, seq_length, d_model):
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

