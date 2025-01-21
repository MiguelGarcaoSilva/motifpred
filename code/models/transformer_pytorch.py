import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series forecasting.
    args:

    input_dim (int): Dimension of the primary input features.
    sequence_length (int): Length of the input sequence.
    d_model (int): Dimension of the model.
    n_heads (int): Number of heads in the multiheadattention models.
    e_layers (int): Number of sub-encoder-layers in the encoder.
    dim_feedforward (int): Dimension of the feedforward network model.
    dropout (float): Dropout value.
    output_dim (int): Dimension of the output features.
    """
    def __init__(
        self,
        input_dim,
        sequence_length,
        d_model,
        n_heads,
        e_layers,
        dim_feedforward,
        dropout,
        output_dim = 1
    ):
        super(TimeSeriesTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(sequence_length, d_model)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # Use batch-first for input shape (batch_size, sequence_length, feature_dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=e_layers)

        # Regression output layer
        self.regression_head = nn.Linear(d_model, output_dim)

    def forward(self, x):

        # Embedding
        x = self.input_embedding(x)  # Shape: (batch_size, sequence_length, d_model)

        # Ensure positional encoding is on the same device as x
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Transformer encoder (no need to permute due to batch_first=True)
        x = self.transformer_encoder(x)  # Shape: (batch_size, sequence_length, d_model)

        # Take the representation of the last time step
        x = x[:, -1, :]  # Shape: (batch_size, d_model)

        # Regression output
        output = self.regression_head(x)  # Shape: (batch_size, output_dim)

        return output


    def _generate_positional_encoding(self, sequence_length, d_model):
        position = torch.arange(0, sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(sequence_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

