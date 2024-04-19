"""This script contains code to train and evaluate a GNN model for edge prediction."""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch_geometric.transforms as T
from tqdm import tqdm

from gnn.data import gen_data
from gnn.GAE import GAEModel
from gnn.GAT_encoder import GATEncoder
from gnn.linear_decoder import LinearEdgeDecoder
from gnn.utils import test, train

# Args
data_filepath = 'Data/eg_data.json'
with_labels = False
num_epochs = 20
encoder_hidden_channels = [128, 64, 32]
heads = 8
encoder_dropout_prob = [0.4, 0.3, 0.2]
# encoder_hidden_channels = [128, 64]
# encoder_dropout_prob = [0.5, 0.2]
encoder_out_channels = 8
decoder_hidden_channels = [encoder_out_channels * 2, 32]
# encoder_hidden_channels = [64, 32]
# decoder_hidden_channels = [64, 32]
lr = 1e-3
num_val = 0.1
num_test = 0.1
random_seed = 1505

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_dir = Path(f'{__file__}').parent / Path('models')
os.makedirs(model_save_dir, exist_ok=True)
model_name = Path(f'SageConv_eg_{datetime.now().strftime("%Y_%m_%d_%H%M")}.pt')
model_save_path = model_save_dir / model_name


def main():
    if random_seed is not None:
        torch.manual_seed(random_seed)

    with open(data_filepath, 'r') as f:
        data_dict = json.load(f)

    data = gen_data(data_dict, with_labels=with_labels)

    # split the data
    train_data, val_data, test_data = T.RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        neg_sampling_ratio=0.3,
        add_negative_train_samples=True,
        is_undirected=True
    )(data)
    print(f'{train_data=}')
    print(f'{val_data=}')
    print(f'{test_data=}')
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    encoder = GATEncoder(hidden_channels=encoder_hidden_channels,
                         out_channels=encoder_out_channels,
                         heads=heads,
                         dropout_probs=encoder_dropout_prob)
    decoder = LinearEdgeDecoder(hidden_channels=decoder_hidden_channels)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    model = GAEModel(
        encoder=encoder,
        decoder=decoder
    )
    model = model.to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    pbar = tqdm(range(num_epochs), desc='Training', unit="Epoch")
    for epoch in pbar:
        loss = train(model, train_data, optimizer, with_labels=with_labels)
        train_rmse = test(model, train_data, with_labels=with_labels)
        val_rmse = test(model, val_data, with_labels=with_labels)
        pbar.set_postfix(ordered_dict={"Loss": f"{loss: .4f}",
                                       "Train": f"{train_rmse: .4f}",
                                       "Val": f"{val_rmse:.4f}"})
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
        #       f'Val: {val_rmse:.4f}')

    # Eval
    test_rmse, pred = test(model, test_data, return_pred=True)
    pred = pred.numpy()
    target = test_data.cpu().edge_label.squeeze(-1).numpy()

    # Save model
    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()
