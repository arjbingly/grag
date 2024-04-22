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
from gnn.utils import EarlyStopping, SaveBestModel, test, train

# Args
# Reproducibility conf
random_seed = 1505
model_name = Path('SageConv_cranfield_data')
# Data conf
data_filepath = 'Data/cranfield.json'
with_labels = True
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
early_stop_patience = 15
early_stop_delta = 0.0001

model_save_dir = Path(f'{__file__}').parent / Path('models')
os.makedirs(model_save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    if with_labels:
        criterion = torch.nn.MSELoss()


        def pre_process(x):
            return torch.clamp(x, -1, 1)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        pre_process = None

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
            'early_stop_patience': early_stop_patience,
            'early_stop_delta': early_stop_delta,
        },
    }

    saver = SaveBestModel(save_dir=model_save_dir, model_name=model_name)
    early_stopper = EarlyStopping(patience=early_stop_patience, delta=early_stop_delta)

    # Training
    pbar = tqdm(range(num_epochs), desc='Training', unit="Epoch")
    train_start_time = datetime.now()
    for epoch in pbar:
        if early_stopper.stop:
            break
        loss = train(model, train_data, optimizer, criterion)
        train_loss = test(model, train_data, criterion, pre_process_func=pre_process)
        val_loss = test(model, val_data, criterion, pre_process_func=pre_process)
        pbar.set_postfix(ordered_dict={"Loss": f"{loss: .4f}",
                                       "Train": f"{train_loss: .4f}",
                                       "Val": f"{val_loss:.4f}"})
        # Save Best Model and config
        train_end_time = datetime.now()
        model_conf['train_conf'] = {
            'start_time': f'{train_start_time}',
            'end_time': f'{train_end_time}',
            'time_taken': f'{train_end_time - train_start_time}',
            'epochs': epoch,
        }
        saver(model, optimizer, criterion, val_loss, model_conf)
        # Early Stopping
        early_stopper(val_loss)

    # Eval
    model.load_state_dict(torch.load(saver.best_model_path)['model_state_dict'])
    train_loss = test(model, train_data, criterion, pre_process_func=pre_process)
    val_loss = test(model, val_data, criterion, pre_process_func=pre_process)
    test_loss = test(model, test_data, criterion, pre_process_func=pre_process)
    print('*** Best Model ***')
    print(f' Train loss: {train_loss:.4f}')
    print(f' Val   loss: {val_loss:.4f}')
    print(f' Test  loss: {test_loss:.4f}')
