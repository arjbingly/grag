"""This script contains code to train and evaluate a GNN model for edge prediction."""

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch_geometric.transforms as T
from tqdm import tqdm

from gnn.data import gen_data
from gnn.sageconv_model import Model
from gnn.utils import test, train

# Args
data_filepath = 'Data/eg_data.json'
num_epochs = 100
hidden_channels = 32
lr = 1e-3
num_val = 0.1
num_test = 0.1
random_seed = 1505

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save_dir = Path(f'{__file__}').parent / Path('models')
os.makedirs(model_save_dir, exist_ok=True)
model_name = Path(f'SageConv_eg_{datetime.now().strftime("%Y_%m_%d_%H%M")}')
model_save_path = model_save_dir / model_name


def main():
    if random_seed is not None:
        torch.manual_seed(random_seed)

    with open(data_filepath, 'r') as f:
        data_dict = json.load(f)

    data = gen_data(data_dict)

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

    model = Model(hidden_channels=hidden_channels).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Training
    pbar = tqdm(range(num_epochs), desc='Training', unit="Epoch")
    for epoch in pbar:
        loss = train(model, train_data, optimizer)
        train_rmse = test(model, train_data)
        val_rmse = test(model, val_data)
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
    # texts = []
    # for data in data_dict.values():
    #     texts.append(data['text'])
    # 
    # doc_index_0 = test_data.edge_label_index[0].cpu().numpy()
    # doc_index_1 = test_data.edge_label_index[1].cpu().numpy()
    # 
    # df = pd.DataFrame({'doc_index_0': doc_index_0, 'doc_index_1': doc_index_1, 'pred': pred, 'target': target})
    # print(df)
    # 
    # doc_0 = []
    # for doc_index in df.doc_index_0:
    #     doc_0.append(texts[doc_index])
    # doc_1 = []
    # for doc_index in df.doc_index_1:
    #     doc_1.append(texts[doc_index])
    # 
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_colwidth', None)
    # df = pd.DataFrame({'doc_0': doc_0, 'doc_1': doc_1, 'pred': pred, 'target': target})
    # print(df)


if __name__ == '__main__':
    main()
