from typing import List, Union

import torch


class LinearEdgeDecoder(torch.nn.Module):
    """Edge decoder module.

    This module predicts edge labels based on node embeddings.
        
    Args:
        hidden_channels (Union[List[int], int]): Number of hidden channels for each linear layer.
            If an integer is provided, a single linear layer will be created with the specified number of channels.
            If a list is provided, multiple linear layers will be created with the number of channels specified
            in the list.

    Attributes:
        lin_layers (List[torch.nn.Linear]): List of linear layers used.
        
    Example:
        >>> decoder = LinearEdgeDecoder(hidden_channels=[128, 64])
    """

    def __init__(self, hidden_channels: Union[List[int], int, None]):
        """Initialize the LinearEdgeDecoder module.
        
        Args:
            hidden_channels (Union[List[int], int]): Number of hidden channels for each convolutional layer.
        """
        super().__init__()
        self.lin_layers = torch.nn.ModuleList()
        if isinstance(hidden_channels, int):
            self.lin_layers.append(torch.nn.Linear(hidden_channels, 1))
        elif isinstance(hidden_channels, list):
            for hidden_channel, out_channel in zip(hidden_channels[:-1], hidden_channels[1:]):
                self.lin_layers.append(torch.nn.Linear(hidden_channel, out_channel))
        self.lin_layers.append(torch.nn.Linear(hidden_channels[-1], 1))

    def forward(self, x, edge_label_index):
        """Forward pass of the decoder module.

        Args:
            x (Tensor): Input features (node embeddings).
            edge_label_index (Tensor): Tuple containing row and column indices of edges.

        Returns:
            Tensor: Predicted edge labels.
        """
        row, col = edge_label_index
        x = torch.cat([x[row], x[col]], dim=-1)
        for layer in self.lin_layers[:-1]:
            x = layer(x)
            x = x.relu()
        x = self.lin_layers[-1](x)
        return x.view(-1)
