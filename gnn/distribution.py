import json

import matplotlib.pyplot as plt

from gnn.data import gen_edges

data_filepath = 'Data/cranfield.json'

with open(data_filepath, 'r') as f:
    data_dict = json.load(f)

embeddings = []
for data in data_dict.values():
    embeddings.append(data['embedding'])

edges, edge_features = gen_edges(embeddings, threshold=0)

plt.hist(edge_features, bins=30, density=True, cumulative=True)
plt.grid()
plt.show()

plt.hist(edge_features, bins=30, density=True)
plt.grid()
plt.show()
