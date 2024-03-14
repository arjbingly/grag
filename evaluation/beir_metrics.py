from hashlib import md5
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import recall_score, precision_score, f1_score, ndcg_score
from tqdm import tqdm

from src.components.multivec_retriever import Retriever
from src.components.utils import get_config

# %%

COLLECTION_NAME = "hotpotqa"
HF_DATASET_NAME = "BeIR/hotpotqa-generated-queries"
TOP_K = 10

config_path = Path(__file__).parent / 'config.ini'
config = get_config(config_path)
config['chroma']["collection_name"] = COLLECTION_NAME

retriever = Retriever(top_k=TOP_K, **config['multivec_retriever'], chroma_kwargs=config['chroma'])


def load_hf_dataset(dataset_name, split=False):
    if split:
        # TODO
        pass
    else:
        return load_dataset(dataset_name)['train']


dataset = load_hf_dataset(HF_DATASET_NAME)

retrieved_chunks = []  # hashes
ground_truth_chunks = []  # hashes
for n, row in enumerate(tqdm(dataset, desc="Retrieving chunks")):
    query = row['query'].strip()
    retrieved_docs = retriever.get_chunk(query)
    _retrieved_chunks = []
    for chunk in retrieved_docs:
        _hash = md5(chunk.page_content.encode()).digest()
        _retrieved_chunks.append(_hash)
    retrieved_chunks.append(_retrieved_chunks)
    if n >= 2:
        break

for n, row in enumerate(tqdm(dataset, desc="Getting ground truth chunks")):
    _hash = md5(row['text'].encode()).digest()
    ground_truth_chunks.append([_hash])
    if n >= 2:
        break
# %%
# compare
# ASSUMPTION: TOP_K > len(ground_truth)
ground_truth_chunks = np.array(ground_truth_chunks)
retrieved_chunks = np.array(retrieved_chunks)
y_true = []
y_pred = []
for i, ground_truth in enumerate(ground_truth_chunks):
    _y_true = np.zeros(TOP_K)
    _y_true[:len(ground_truth)] = 1
    mask = (retrieved_chunks[i][:, np.newaxis] == ground_truth)
    _y_pred = np.zeros(TOP_K)
    _y_pred[mask.any(axis=1)] = 1
    y_true.append(_y_true)
    y_pred.append(_y_pred)
# %%
precision = precision_score(y_true, y_pred, average='micro')
recall = recall_score(y_true, y_pred, average='micro')
f1 = f1_score(y_true, y_pred, average='micro')
ndcg = ndcg_score(y_true, y_pred)
