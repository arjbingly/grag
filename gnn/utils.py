"""This module contains utility functions."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from grag.components.embedding import Embedding
from tqdm import tqdm


def gen_embeddings(data, verbose=True):
    """Generates embeddings for a list of sentences using a specified embedding model.

    Args:
        verbose: 
        data (list): A list of strings containing sentences to generate embeddings for.

    Returns:
        list: A list of embeddings corresponding to the input sentences.

    Example:
        >>> sentences = ["This is the first sentence.", "This is the second sentence."]
        >>> embeddings = gen_embeddings(sentences)
    """
    embedding = Embedding("instructor-embedding", "hkunlp/instructor-xl")
    embedding_instruction = "Represent the sentence for retrieval"
    embedding.embedding_function.query_instruction = embedding_instruction
    embedding_func = embedding.embedding_function.embed_query

    if verbose:
        embeddings = []
        for string in tqdm(data, desc="Generating embeddings"):
            embeddings.append(embedding_func(string))
    else:
        embeddings = [embedding_func(string) for string in data]
    return embeddings


def cosine_similarity(a, b):
    """Computes the cosine similarity between two vectors.

    Args:
        a (numpy.ndarray): The first vector.
        b (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between vectors 'a' and 'b'.

    Example:
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([4, 5, 6])
        >>> similarity = cosine_similarity(a, b)
    """
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


def train(model, data, optimizer, criterion):
    """Trains the given model on the provided data.

    Args:
        model (torch.nn.Module): The model to train.
        data (Data): A PyTorch Geometric Data object representing the graph data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.

    Returns:
        float: The loss value after training.

    Example:
        >>> model = MyGraphModel()
        >>> data = gen_data(data_dict)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> loss = train(model, data, optimizer)
    """
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index, data.edge_label_index)
    target = data.edge_label.squeeze(-1)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, data, metric, pre_process_func=None, return_pred=False):
    """Evaluates the given model on the provided data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (Data): A PyTorch Geometric Data object representing the graph data.
        return_pred (bool, optional): Whether to return the predictions. Defaults to False.

    Returns:
        float or tuple: If return_pred is False, returns the root mean squared error (RMSE) of the model's predictions. 
                        If return_pred is True, returns a tuple containing the RMSE and the predictions.

    Example:
        >>> model = MyGraphModel()
        >>> data = gen_data(data_dict)
        >>> rmse = test(model, data)
        >>> rmse, predictions = test(model, data, return_pred=True)
    """
    model.eval()
    target = data.edge_label.squeeze(-1)
    pred = model(data.x, data.edge_index, data.edge_label_index)
    if pre_process_func:
        pred = pre_process_func(pred.detach())
    score = metric(pred, target)
    pred = pred.cpu()
    if return_pred:
        return score, pred
    else:
        return score


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001, is_loss=True, verbose=True, ask_user=True, pbar=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.is_loss = is_loss
        self.stop = False
        self.verbose = verbose
        self.ask_user = ask_user
        self.pbar = pbar

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.stop = False

    def prompt_user(self):
        invalid_input = True
        while invalid_input:
            user_input = input("Do you still want to continue training? [Yes(Y)/No(N)]")
            if user_input.lower().strip()[0] == 'y':
                self.reset()
                invalid_input = False
            elif user_input.lower().strip()[0] == 'n':
                invalid_input = False

    def step(self, score):
        score = score.detach().cpu().item()
        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        if self.is_loss:
            if score > self.best_score + self.delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score < self.best_score - self.delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0

    def __call__(self, score):
        self.step(score)
        if self.counter >= self.patience:
            self.stop = True
            if self.verbose:
                if self.pbar:
                    self.pbar.write(f"Epoch {self.pbar.n}:Early stopping.")
                else:
                    print('Early stopping...')
            if self.ask_user:
                self.prompt_user()


class SaveBestModel:
    def __init__(self, save_dir='models', is_loss=True, model_name=None, score_name='valid_loss', pbar=None):
        self.is_loss = is_loss
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.best_score = None
        self.score_name = score_name
        self.best_model_path = None
        self.id = datetime.now().strftime("%Y_%m_%d_%H%M")
        self.pbar = pbar

    def save(self, model, optimizer, criterion, conf_json=None):
        save_path = self.save_dir / f'{self.model_name}_{self.id}'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, f'{save_path}.pt')
        if conf_json is not None:
            with open(f'{save_path}.json', 'w+') as f:
                json.dump(conf_json, f, indent=4)
        self.best_model_path = f'{save_path}.pt'

    def __call__(self, model, optimizer, criterion, score, conf_json=None):
        score = score.detach().cpu().item()
        if self.best_score is None:
            self.best_score = score
        if self.is_loss:
            if score < self.best_score:
                if self.pbar:
                    self.pbar.write(f'Epoch-{self.pbar.n}: New best {self.score_name}:{score}, saving model.')
                else:
                    print(f'New best {self.score_name}:{score}, saving model.')
                self.best_score = score
                self.save(model, optimizer, criterion, conf_json)
        else:
            if score > self.best_score:
                if self.pbar:
                    self.pbar.write(f'Epoch-{self.pbar.n}: New best {self.score_name}:{score}, saving model.')
                else:
                    print(f'New best {self.score_name}:{score}, saving model.')
                self.best_score = score
                self.save(model, optimizer, criterion, conf_json)
