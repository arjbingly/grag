import time
import numpy as np
import asyncio
from tqdm import tqdm

# add code folder to sys path
import os
from pathlib import Path
import sys
sys.path.insert(1, str(Path(os.getcwd()).parents[0]))


from utils.txt_data_ingest import load_split_dir
from components.chroma_client import ChromaClient

data_path = Path(os.getcwd()).parents[1]/'data'/'Gutenberg'/'txt' #"data/Gutenberg/txt"

def main():
    docs = load_split_dir(data_path)
    print(f'Total Number of docs: {len(docs)}')

    client = ChromaClient()

    print('Before Adding Docs...')
    print(f'The {client.collection_name} has {client.collection.count()} documents')

    n_docs = 100
    n_steps = 10
    if n_steps * n_docs > len(docs):
        raise Exception('Not enough docs')

    sync_times = []
    async_times = []

    for step in tqdm(range(n_steps), desc='Sync test'):
        start = time.perf_counter()
        client.add_docs(docs[step:n_docs * step])
        sync_times.append(time.perf_counter() - start)

    print(f'Avg Sync times {np.mean(sync_times)}')
    print(f'Min Sync time {np.min(sync_times)}')
    print(f'Max Sync time {np.max(sync_times)}')

    for step in tqdm(range(n_steps), desc='Async test'):
        start = time.perf_counter()
        asyncio.run(client.aadd_docs(docs[step:n_docs * step]))
        async_times.append(time.perf_counter() - start)

    print(f'Avg Async times {np.mean(async_times)}')
    print(f'Min Async time {np.min(async_times)}')
    print(f'Max Async time {np.max(async_times)}')

    print('After Adding Docs...')
    print(f'The {client.collection_name} has {client.collection.count()} documents')


if __name__ == "__main__":

    main()



