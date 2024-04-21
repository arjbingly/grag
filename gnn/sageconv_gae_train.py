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
from gnn.linear_decoder import LinearEdgeDecoder
from gnn.sageconv_encoder import SageConvEncoder
from gnn.utils import test, train

# Args
# Reproducibility conf
random_seed = 1505
# Data conf
data_filepath = 'Data/cranfield.json'
with_labels = False
# Random Link Split conf
num_val = 0.1
num_test = 0.1
neg_sampling_ratio = 0.3
# Model conf
encoder_hidden_channels = [128, 64]
encoder_dropout_prob = [0.5, 0.2]
encoder_out_channels = 32
decoder_hidden_channels = [encoder_out_channels * 2, 32]
# Training conf
lr = 1e-3
num_epochs = 3000
early_stop_n_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_dir = Path(f'{__file__}').parent / Path('models')
os.makedirs(model_save_dir, exist_ok=True)
model_name = Path(f'SageConv_cranfield_{datetime.now().strftime("%Y_%m_%d_%H%M")}')
model_save_path = model_save_dir / f'{model_name}.pt'
config_save_path = model_save_dir / f'{model_name}.json'

if __name__ == '__main__':
    if random_seed is not None:
        torch.manual_seed(random_seed)

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
    print(f'{train_data=}')
    print(f'{val_data=}')
    print(f'{test_data=}')
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    encoder = SageConvEncoder(hidden_channels=encoder_hidden_channels,
                              out_channels=encoder_out_channels,
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

    train_start_time = datetime.now()
    for epoch in pbar:
        loss = train(model, train_data, optimizer, with_labels=with_labels)
        train_rmse = test(model, train_data, with_labels=with_labels)
        val_rmse = test(model, val_data, with_labels=with_labels)
        pbar.set_postfix(ordered_dict={"Loss": f"{loss: .4f}",
                                       "Train": f"{train_rmse: .4f}",
                                       "Val": f"{val_rmse:.4f}"})
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
        #       f'Val: {val_rmse:.4f}')
        # Early Stopping
        if epoch == 0:
            prev_val_rmse = val_rmse.detach().cpu().numpy()
            early_stop_epochs = 0
        else:
            if prev_val_rmse <= val_rmse.detach().cpu().numpy():
                early_stop_epochs += 1
                if early_stop_epochs >= early_stop_n_epochs:
                    print(f'Early stopping at epoch {epoch}')
                    break
            else:
                early_stop_epochs = 0
                prev_val_rmse = val_rmse.detach().cpu().numpy()
    train_end_time = datetime.now()
    # Eval
    test_rmse, pred = test(model, test_data, return_pred=True)
    pred = pred.numpy()
    target = test_data.cpu().edge_label.squeeze(-1).numpy()

    # Save model and config
    torch.save(model.state_dict(), model_save_path)

    model_conf = {
        'reproducibility_conf': {
            'random_seed': random_seed,
        },
        'data_conf': {
            'data_filepath': data_filepath,
            'with_labels': with_labels,
        },
        'data_split_conf': {
            'num_val': num_val,
            'num_test': num_test,
            'neg_sampling_ratio': neg_sampling_ratio,
        },
        'model_conf': {
            'model': f'{model}',
            'encoder_hidden_channels': encoder_hidden_channels,
            'encoder_dropout_prob': encoder_dropout_prob,
            'encoder_out_channels': encoder_out_channels,
            'decoder_hidden_channels': decoder_hidden_channels,
        },
        'train_conf': {
            'num_epochs': num_epochs,
            'lr': lr,
            'early_stop_n_epochs': early_stop_n_epochs,
        },
        'train_log': {
            'start_time': f'{train_start_time}',
            'end_time': f'{train_end_time}',
            'time_taken': f'{train_end_time - train_start_time}',
            'epochs': epoch,
        }
    }
    with open(config_save_path, 'w+') as f:
        json.dump(model_conf, f, indent=4)
