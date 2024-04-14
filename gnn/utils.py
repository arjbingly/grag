"""This module contains utility functions."""

import numpy as np
import torch
import torch.nn.functional as F


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


def train(model, data, optimizer):
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
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, data, return_pred=False):
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
    pred = model(data.x, data.edge_index, data.edge_label_index)
    pred = pred.clamp(min=-1, max=1)
    target = data.edge_label.squeeze(-1)
    rmse = F.mse_loss(pred, target).sqrt()
    pred = pred.cpu()
    if return_pred:
        return rmse, pred
    else:
        return rmse
