"""This module contains classes for a SAGE Convolutional Graph Neural Network for edge prediction."""

import torch
from torch_geometric.nn import SAGEConv


class GNNEncoder(torch.nn.Module):
    """Graph Neural Network (GNN) encoder module.

    This module applies a series of graph convolutional operation to the input features.

    Args:
        hidden_channels (int): Number of hidden channels in the encoder.
        out_channels (int): Number of output channels in the encoder.

    Example:
        >>> encoder = GNNEncoder(hidden_channels=64, out_channels=32)
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=(-1, -1), out_channels=hidden_channels)
        self.conv2 = SAGEConv(in_channels=(-1, -1), out_channels=out_channels)

    def forward(self, x, edge_index):
        """Forward pass of the encoder module.

        Args:
            x (Tensor): Input features (node features).
            edge_index (LongTensor): Graph edge indices.

        Returns:
            Tensor: Output features after applying graph convolution.
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    """Edge decoder module.

    This module predicts edge labels based on node embeddings.

    Args:
        hidden_channels (int): Number of hidden channels in the decoder.

    Example:
        >>> decoder = EdgeDecoder(hidden_channels=64)
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

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
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        return x.view(-1)


class Model(torch.nn.Module):
    """Graph neural network model based on SAGEConv for edge prediction.

    This model consists of a SAGEConv based GNN encoder followed by an edge decoder.

    Args:
        hidden_channels (int): Number of hidden channels in the encoder/decoder.

    Example:
        >>> model = Model(hidden_channels=64)
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x, edge_index, edge_label_index):
        """Forward pass of the model.

        Args:
            x (Tensor): Input features (node features).
            edge_index (Tensor): Graph edge indices.
            edge_label_index (Tensor): Tuple containing row and column indices of edges.

        Returns:
            Tensor: Predicted edge labels.
        """
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_label_index)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(hidden_channels=5).to(device)
    print(model)
