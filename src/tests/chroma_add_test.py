import asyncio
# add code folder to sys path
import os
from pathlib import Path

from grag.components.chroma_client import ChromaClient

data_path = Path(os.getcwd()).parents[1] / 'data' / 'Gutenberg' / 'txt'  # "data/Gutenberg/txt"


def main():
    docs = load_split_dir(data_path)
    # print(f'Total Number of docs: {len(docs)}')

    client = ChromaClient()

    print('Before Adding Docs...')
    print(f'The {client.collection_name} has {client.collection.count()} documents')

    n_docs = 10
    print(f'Adding {n_docs // 2} docs synchronously')
    client.add_docs(docs[:n_docs // 2])
    print(f'Adding {n_docs // 2} docs asynchronously')
    asyncio.run(client.aadd_docs(docs[n_docs // 2:]))

    print('After Adding Docs...')
    print(f'The {client.collection_name} has {client.collection.count()} documents')


if __name__ == "__main__":
    main()
    print('All Tests Passed')
