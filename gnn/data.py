"""This module contains functions to create torch_geometric Data objects from a JSON file."""

from itertools import combinations

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from tqdm import tqdm

from gnn.utils import cosine_similarity


def gen_egdes(embeddings, verbose=True, threshold=0.5):
    """Generates edges and edge features for a graph based on pairwise cosine similarity of embeddings.

    Args:
        verbose: 
        embeddings (list): A list of embeddings for each document.
        verbose (bool, optional): If True, prints out the edges and edge features for each document. Defaults to True.
        threshold (float, optional): Threshold for cosine similarity. Defaults to 0.5.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - edges: An array of shape (2, num_edges) containing the indices of connected nodes.
            - edge_features: An array of shape (num_edges, 1) containing the cosine similarities between connected nodes.
            

    Example:
        >>> embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), np.array([0.7, 0.8, 0.9])]
        >>> edges, edge_features = gen_egdes(embeddings)
    """
    edges = []
    edge_features = []
    if verbose:
        pbar = tqdm(combinations(enumerate(embeddings), 2))
    else:
        pbar = combinations(enumerate(embeddings), 2)
    for doc1, doc2 in pbar:
        similarity = cosine_similarity(doc1[1], doc2[1])
        if similarity > threshold:
            edges.append((doc1[0], doc2[0]))
            edge_features.append([similarity])
    edges = np.array(edges).T
    edge_features = np.array(edge_features)
    return edges, edge_features


def gen_data(data_dict, verbose=True):
    """Generates an undirected graph data object from a dictionary of embeddings.
    
    The nodes are text embeddings, and edges are cosine similarities.

    Args:
       data_dict (dict): A dictionary where keys are document indices and values are dictionaries containing 'embedding' key.
       verbose (bool, optional): Whether to print information about the generated data. Defaults to True.

    Returns:
       Data: A PyTorch Geometric Data object representing the graph.

    Example:
       >>> data_dict = {
       ...     0: {'embedding': array([0.1, 0.2, 0.3])},
       ...     1: {'embedding': array([0.4, 0.5, 0.6])}
       ... }
       >>> graph_data = gen_data(data_dict)
    """
    embeddings = []
    for data in data_dict.values():
        embeddings.append(data['embedding'])

    edges, edge_features = gen_edges(embeddings)

    x = torch.tensor(embeddings, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_label = torch.tensor(edge_features, dtype=torch.float)

    if verbose:
        print(f'{x.shape=}')
        print(f'{edge_index.shape=}')
        print(f'{edge_label.shape=}')

    data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
    data.validate(raise_on_error=True)
    data = T.ToUndirected()(data)
    # ToUndirected adds reverse edges with the same features -?

    if verbose:
        print(f'{data=}')

    return data


if __name__ == '__main__':
    data = gen_data('data.json')
