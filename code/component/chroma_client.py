from typing import List
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document

from config import chroma


# %%
class ChromaClient:
    def __init__(self):
        self.host = chroma['host']
        self.port = chroma['port']
        self.embedding_model = chroma['embedding_model']
        self.collection_name = chroma['collection_name']

        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        self.chroma_client = chromadb.HttpClient(host=self.host, port=self.port)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        self.langchain_chroma = Chroma(client=self.chroma_client,
                                       collection_name=self.collection_name,
                                       embedding_function=self.embedding_function, )

    def test_connection(self):
        if self.chroma_client.heartbeat():
            print(f'Connection to {self.host}/{self.port} is alive..')
        else:
            print(f'Connection to {self.host}/{self.port} is not alive !!')

    async def aadd_docs(self,docs: List[Document], verbose=True):
        tasks = [self.langchain_chroma.aadd_documents([doc]) for doc in docs]
        if verbose:
            await tqdm_asyncio.gather(*tasks)
        else:
            await asyncio.gather(*tasks)

    def add_docs(self,docs: List[Document], verbose=True):
        for doc in (tqdm(docs, desc='Adding Documents:') if verbose else docs):
            _id = self.langchain_chroma.add_documents([doc])


if __name__ == "__main__":
    client = ChromaClient()
    client.test_connection()
