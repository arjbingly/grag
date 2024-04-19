"""This module contains classes for a SAGE Convolutional Graph Neural Network for edge prediction."""
from itertools import zip_longest
from typing import List, Union

import torch
from torch_geometric.nn import SAGEConv


class SageConvEncoder(torch.nn.Module):
    """Sage Convolution based Graph Network Encoder.

    This module is responsible for encoding graph-structured data using Sage Convolutional based Graph Neural Networks (GNNs).

    Args:
        hidden_channels (Union[List[int], int]): Number of hidden channels for each convolutional layer.
            If an integer is provided, a single convolutional layer will be created with the specified number of channels.
            If a list is provided, multiple convolutional conv_layers will be created with the number of channels specified
            in the list.
        out_channels (int): Number of output channels for the final convolutional layer.
        
    Example:
        >>> encoder = SageConvEncoder(hidden_channels=[64,32,16], out_channels=16, dropout_porb = 0.2)
        This creates an encoder network with 3 hidden conv_layers each of which has 64, 32, and 16 hidden chanells respectively.
        The final layer has 16 channels.
    """

    def __init__(self,
                 hidden_channels: Union[List[int], int, None],
                 out_channels: int,
                 dropout_probs: Union[List[float], float, None] = 0.2):
        """Initialize the SageConvEncoder module.

        Args:
            hidden_channels (Union[List[int], int]): Number of hidden channels for each convolutional layer.
            out_channels (int): Number of output channels for the final convolutional layer.
            dropout_probs (float): Dropout probability.

        """
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        if isinstance(hidden_channels, int):
            self.conv_layers.append(SAGEConv(in_channels=(-1, -1), out_channels=hidden_channels))
        elif isinstance(hidden_channels, list):
            for hidden_channel in hidden_channels:
                self.conv_layers.append(SAGEConv(in_channels=(-1, -1), out_channels=hidden_channel))
        self.conv_layers.append(SAGEConv(in_channels=(-1, -1), out_channels=out_channels))

        self.dropout_layers = torch.nn.ModuleList()
        if isinstance(dropout_probs, float):
            for i in range(len(self.conv_layers) - 1):
                self.dropout_layers.append(torch.nn.Dropout(p=dropout_probs))
        elif isinstance(dropout_probs, list):
            for dropout_prob in dropout_probs:
                self.dropout_layers.append(torch.nn.Dropout(p=dropout_prob))

    def forward(self, x, edge_index):
        """Forward pass of the GNNEncoder module.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
            Tensor: The encoded node features.

        """
        for conv_layer, dropout_layer in zip_longest(self.conv_layers[:-1], self.dropout_layers):
            x = conv_layer(x, edge_index)
            if dropout_layer is not None:
                x = dropout_layer(x)
            x = x.relu()
        x = self.conv_layers[-1](x, edge_index)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = SageConvEncoder(hidden_channels=[128, 64], out_channels=64)
    print(encoder)
