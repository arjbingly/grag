import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.utils.convert import to_networkx

from data import gen_data
from gnn.sageconv_model import Model

hidden_channels = 32

model_path = Path(__file__).parent / 'models' / 'SageConv_eg_2024_04_14_2057.pt'
gnn_model = Model(hidden_channels=hidden_channels)
gnn_model.load_state_dict(torch.load(model_path))
gnn_model.eval()

explainer = Explainer(gnn_model, GNNExplainer(), explanation_type='model',
                      model_config=ModelConfig(mode='regression', task_level='edge', return_type='raw'),
                      edge_mask_type='object')
lab = explainer.get_prediction(gnn_model)

with open('Data/eg_data.json', 'r') as f:
    data_dict = json.load(f)

pyg_data = gen_data(data_dict)
G = to_networkx(pyg_data, edge_attrs=['edge_label'], to_undirected=True)

edge_lables = nx.get_edge_attributes(G, 'edge_label')  # Get edge labels
# format edge_labels
for edge_index, edge_label in edge_lables.items():
    edge_lables[edge_index] = round(edge_label[0], 3)

plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, pos=pos)
nx.draw_networkx_edge_labels(G, edge_labels=edge_lables, pos=pos)
plt.show()
