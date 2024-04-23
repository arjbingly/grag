import json

import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

from gnn.data import gen_data

data_filepath = 'Data/eg_data.json'
with_labels = True
# Random Link Split conf
num_val = 0.1
num_test = 0.1
neg_sampling_ratio = 0.3
with open(data_filepath, 'r') as f:
    data_dict = json.load(f)

data = gen_data(data_dict, with_labels=with_labels)

# split the data
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=num_val,
    num_test=num_test,
    neg_sampling_ratio=neg_sampling_ratio,
    add_negative_train_samples=True,
    is_undirected=True
)(data)

loader = NeighborLoader(
    train_data,
    # Sample 30 neighbors for each node and edge type for 2 iterations
    num_neighbors=[5] * 2,
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=128,
    # input_nodes=('paper', train_data['paper'].train_mask),
)

sampled_hetero_data = next(iter(loader))
print(sampled_hetero_data)
