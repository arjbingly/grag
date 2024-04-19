import torch


class GAEModel(torch.nn.Module, ):
    """Graph Auto-Encoder neural network model based on SAGEConv for edge prediction.

    This model consists of a SAGEConv based GNN encoder followed by an edge decoder.

    Args:
        hidden_channels (int): Number of hidden channels in the encoder/decoder.

    Example:
        >>> encoder = SageConvEncoder(hidden_channels=64)
        >>> decoder = LinearEdgeDecoder(hidden_channels=64)
        >>> model = GAEModel(encoder=encoder, decoder=decoder)
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

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
