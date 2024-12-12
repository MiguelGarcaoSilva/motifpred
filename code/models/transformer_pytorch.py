import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerX1(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, num_heads, num_layers, dropout):
        """
        Args:
            input_dim: int, number of features in the input
            hidden_sizes: list of ints, sizes of the hidden layers
            output_dim: int, number of features in the output
            num_heads: int, number of attention heads
            num_layers: int, number of transformer layers
            dropout: float, dropout rate
        """
        super(TransformerX1, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_layer = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden_sizes[i], nhead=num_heads) for i in range(len(hidden_sizes) - 1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)
        self.transform