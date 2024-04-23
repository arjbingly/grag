import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from gnn.data import gen_data
from gnn.GAE import GAEModel
from gnn.linear_decoder import LinearEdgeDecoder
from gnn.sageconv_encoder import SageConvEncoder

model_name = 'SageConv_eg_data_2024_04_22_0115'
data_filepath = 'Data/cranfield.json'

model_dir = Path(__file__).resolve().parent / 'models'
model_path = model_dir / f'{model_name}.pt'
model_config_path = model_dir / f'{model_name}.json'

strategy = 'LeaveOneOut'
with_labels = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(data_filepath, 'r') as f:
    data_dict = json.load(f)

data = gen_data(data_dict, with_labels=with_labels)

with open(model_config_path, 'r') as f:
    model_config = json.load(f)

encoder = SageConvEncoder(hidden_channels=model_config['model_conf']['encoder_hidden_channels'],
                          out_channels=model_config['model_conf']['encoder_out_channels'],
                          dropout_probs=model_config['model_conf']['encoder_dropout_prob'])
decoder = LinearEdgeDecoder(hidden_channels=model_config['model_conf']['decoder_hidden_channels'])
encoder = encoder.to(device)
decoder = decoder.to(device)
model = GAEModel(
    encoder=encoder,
    decoder=decoder
)
model = model.to(device)

model.load_state_dict(torch.load(model_path)['model_state_dict'])

# %%
# for node_index_l, node_index_r in zip(data.edge_index[0], data.edge_index[1]):
#     # node_index_l = node_index_l.to(device)
#     # l = (edge_index[0] == node_index_l)
#     # r = (edge_index[1] == node_index_r)
#     # index = torch.logical_and(l, r)
#     break
# %%

# x = data.x.to(device)
# edge_index = data.edge_index.to(device)
# for edge_ind in edge_index:
#     edge_label_index = data.edge_index[:, 0].unsqueeze(1).to(device)
#     with torch.no_grad():
#         pred = model(x, edge_index, edge_label_index)
# %%
# Pass All Strategy - passes the whole grpah to the encoder.
x = data.x.to(device)
edge_index = data.edge_index.to(device)
# Create an edge_index with all possible edges for prediction.
edge_label_index = np.array(list(combinations(range(data.x.shape[0]), 2))).T
edge_label_index = torch.from_numpy(edge_label_index).long().to(device)

with torch.no_grad():
    preds = model(x, edge_index, edge_label_index)
    preds = preds.sigmoid()

# %%
# Leave one out Strategy
x = data.x.to(device)
preds = []
for i, edge in enumerate(data.edge_index.numpy().T):
    edge_index = np.delete(data.edge_index.numpy(), i, axis=1)
    edge_index = torch.from_numpy(edge_index).to(device)
    edge_label_index = data.edge_index[:, i].unsqueeze(1).to(device)
    pred = model(x, edge_index, edge_label_index)
    pred = pred.detach().cpu().item()
    preds.append(pred)
    break
