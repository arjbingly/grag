"""This module contains classes for a SAGE Convolutional Graph Neural Network for edge prediction."""
from typing import List, Union

import torch
from torch_geometric.nn import SAGEConv


class SageConvEncoder(torch.nn.Module):
    """Sage Convolution based Graph Network Encoder.

    This module is responsible for encoding graph-structured data using Sage Convolutional based Graph Neural Networks (GNNs).

    Args:
        hidden_channels (Union[List[int], int]): Number of hidden channels for each convolutional layer.
            If an integer is provided, a single convolutional layer will be created with the specified number of channels.
            If a list is provided, multiple convolutional layers will be created with the number of channels specified
            in the list.
        out_channels (int): Number of output channels for the final convolutional layer.

    Attributes:
        conv_layers (List[SAGEConv]): List of SAGEConv layers used for graph convolution.
        
    Example:
        >>> encoder = SageConvEncoder(hidden_channels=[64,32,16], out_channels=16)
        This creates an encoder network with 3 hidden layers each of which has 64, 32, and 16 hidden chanells respectively.
        The final layer has 16 channels.
    """

    def __init__(self, hidden_channels: Union[List[int], int], out_channels: int):
        """Initialize the SageConvEncoder module.

        Args:
            hidden_channels (Union[List[int], int]): Number of hidden channels for each convolutional layer.
            out_channels (int): Number of output channels for the final convolutional layer.

        """
        super().__init__()
        if isinstance(hidden_channels, int):
            self.conv_layers = [SAGEConv(in_channels=(-1, -1), out_channels=hidden_channels)]
        elif isinstance(hidden_channels, list):
            self.conv_layers = torch.nn.ModuleList()
            for hidden_channel in hidden_channels:
                self.conv_layers.append(SAGEConv(in_channels=(-1, 1), out_channels=hidden_channel))
        self.conv_layers.append(SAGEConv(in_channels=(-1, -1), out_channels=out_channels))

    def forward(self, x, edge_index):
        """Forward pass of the GNNEncoder module.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
            Tensor: The encoded node features.

        """
        for layer in self.conv_layers[:-1]:
            x = layer(x, edge_index)
            x = x.relu()
        x = self.conv_layers[-1](x, edge_index)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = SageConvEncoder(hidden_channels=[128, 64], out_channels=64)
    decoder = LinearEdgeDecoder(hidden_channels=64)
    model = GAEModel(encoder=encoder, decoder=decoder)
    print(model)
