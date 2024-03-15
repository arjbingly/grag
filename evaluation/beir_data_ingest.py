import asyncio
import math
from pathlib import Path

from datasets import load_dataset
from langchain_core.documents import Document
from tqdm.asyncio import tqdm

from src.components.multivec_retriever import Retriever
from src.components.utils import get_config

# %%
DRY_RUN = False
BATCH_SIZE = 2048
COLLECTION_NAME = "hotpotqa"
HF_DATASET_NAME = "BeIR/hotpotqa-generated-queries"

# %%
dataset = load_dataset(HF_DATASET_NAME)['train']

config_path = Path(__file__).parent / 'config.ini'
config = get_config(config_path)
config['chroma']["collection_name"] = COLLECTION_NAME

retriever = Retriever(**config['multivec_retriever'], chroma_kwargs=config['chroma'])


# %%
def batch_gen(dataset, batch_size=128):
    n_samples = len(dataset)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        if end > n_samples:
            yield dataset[start:n_samples]
        yield dataset[start:end]


# %%
# if __name__ == "__main__":
#     print(f'{DRY_RUN =}')
#     print(f'{COLLECTION_NAME =}')
#     print(f'{HF_DATASET_NAME =}')
#     print(f'Number of samples: {len(dataset)}')
#
#     for batch in tqdm(batch_gen(dataset, batch_size=BATCH_SIZE), total=math.ceil(len(dataset) / BATCH_SIZE)):
#         docs = list(map(lambda row: Document(page_content=row[0], metadata={"source": row[1]}),
#                         list(zip(batch['text'], batch['_id']))))
#         if not DRY_RUN:
#             asyncio.run(retriever.aadd_docs(docs, skip_chunking=True))


async def main():
    for batch in tqdm(batch_gen(dataset, BATCH_SIZE), total=math.ceil(len(dataset) / BATCH_SIZE)):
        docs = list(map(lambda row: Document(page_content=row[0], metadata={"source": row[1]}),
                        list(zip(batch['text'], batch['_id']))))
        if not DRY_RUN:
            await retriever.aadd_docs(docs, skip_chunking=True)


if __name__ == '__main__':
    asyncio.run(main())
