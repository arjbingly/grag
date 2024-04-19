from itertools import zip_longest
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv


class GATEncoder(torch.nn.Module):
    """Graph Attention Network."""

    def __init__(self, hidden_channels: Union[List[int], int, None],
                 out_channels: int,
                 heads: Union[List[int], int, None],
                 dropout_probs: Union[List[float], float, None] = 0.2):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        if isinstance(hidden_channels, int):
            self.gat_layers.append(GATv2Conv((-1, -1), out_channels=hidden_channels, heads=heads))
        elif isinstance(hidden_channels, list):
            if isinstance(heads, int):
                heads = [heads for _ in range(len(hidden_channels))]
            assert len(hidden_channels) == len(heads), ValueError(
                f"Certain layers don't have heads, {len(hidden_channels)=} != {len(heads)=}.")
            for hidden_channel, head in zip(hidden_channels, heads):
                self.gat_layers.append(GATv2Conv((-1, -1), hidden_channel, head))
        self.gat_layers.append(GATv2Conv(hidden_channels[-1] * heads[-1], out_channels, heads=1))

        self.dropout_layers = nn.ModuleList()
        if isinstance(dropout_probs, float):
            for i in range(len(self.gat_layers) - 1):
                self.dropout_layers.append(nn.Dropout(p=dropout_probs))
        elif isinstance(dropout_probs, list):
            assert len(dropout_probs) <= len(self.gat_layers) - 1, ValueError(
                f"Number of dropout layers exceed the number of hidden layers, {len(dropout_probs)=} > {len(self.gat_layers) - 1=}.")
            for dropout_prob in dropout_probs:
                self.dropout_layers.append(nn.Dropout(p=dropout_prob))

    def forward(self, h, edge_index, return_attention_weights=False):
        if return_attention_weights is False:
            return_attention_weights = None  # Bug in PyG's code
        for gat_layer, dropout_layer in zip_longest(self.gat_layers[:-1], self.dropout_layers):
            h = gat_layer(h, edge_index)
            if dropout_layer is not None:
                h = dropout_layer(h)
            h = F.elu(h)
        h = self.gat_layers[-1](h, edge_index, return_attention_weights=return_attention_weights)
        return h


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = GATEncoder(hidden_channels=[128, 64, 32], out_channels=64, heads=[8, 4, 2], dropout_probs=[0.4, 0.2])
    print(encoder)
