from typing import List
import asyncio
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from .config import chroma_conf

class ChromaClient:
    def __init__(self):
        self.host = chroma_conf['host']
        self.port = chroma_conf['port']
        self.embedding_type = chroma_conf['embedding_type']
        self.embedding_modelname = chroma_conf['embedding_model']
        self.collection_name = chroma_conf['collection_name']

        match self.embedding_type:
            case 'sentence-transformers':
                self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_modelname)
            case 'instructor-embedding':
                self.embedding_instuction = 'Represent the document for retrival'
                self.embedding_function = HuggingFaceInstructEmbeddings(model_name = self.embedding_modelname)
                self.embedding_function.embed_instruction = self.embedding_instuction
            case _ :
                raise Exception('conifg:embedding_model is invalid')

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
            await tqdm_asyncio.gather(*tasks, desc=f'Adding to {self.collection_name}')
        else:
            await asyncio.gather(*tasks)

    def add_docs(self,docs: List[Document], verbose=True):
        for doc in (tqdm(docs, desc=f'Adding to {self.collection_name}:') if verbose else docs):
            _id = self.langchain_chroma.add_documents([doc])
