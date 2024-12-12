import torch
from torch import nn
import torch.nn.functional as F


class FFNNX1(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim):
        super(FFNNX1, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_sizes_list[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]) for i in range(len(hidden_sizes_list) - 1)])
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, windown_len * features)

        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)


class FFNNX1_X2Masking(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim):
        super(FFNNX1_X2Masking, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_sizes_list[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]) for i in range(len(hidden_sizes_list) - 1)])
        self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

    def forward(self, x, mask):
        # X1 is size (batch_size, windown_len, features)
        # mask_input is size (batch_size, windown_len)
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, windown_len * features)

        #concat mask to input
        x = torch.cat((x, mask), dim = 1)

        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)


#TODO: correct this if needed to use, needs to have the auxiliary input
# class FFNNX1_X2Indices(nn.Module):
#     def __init__(self, input_dim, hidden_sizes_list, output_dim):
#         super(FFNNX1_X2Indices, self).__init__()

#         self.input_layer = nn.Linear(input_dim, hidden_sizes_list[0])
#         self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes_list[i], hidden_sizes_list[i + 1]) for i in range(len(hidden_sizes_list) - 1)])
#         self.output_layer = nn.Linear(hidden_sizes_list[-1], output_dim)

#     def forward(self, x, indices):
#         # X1 is size (batch_size, windown_len, features)
#         # indices is size (batch_size, max_indices_in_window)
#         x = x.view(x.size(0), -1)  # Reshape to (batch_size, windown_len * features)

#         #concat indices to input
#         x = torch.cat((x, indices), dim = 1)

#         x = F.relu(self.input_layer(x))
#         for hidden_layer in self.hidden_layers:
#             x = F.relu(hidden_layer(x))
#         return self.output_layer(x)

                
