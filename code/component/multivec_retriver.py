from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_core.documents import Document

from chroma_client import ChromaClient
from text_splitter import TextSplitter
from config import multivec_retriver_conf

import os
import uuid
from typing import List
import asyncio

#%%

class Retriver:
    def __init__(self, top_k=1):
        self.store_path = multivec_retriver_conf['store_path']
        self.id_key = multivec_retriver_conf['id_key']
        self.client = ChromaClient()
        self.store = LocalFileStore(self.store_path)
        self.retriever = MultiVectorRetriever(
                        vectorstore=self.client.langchain_chroma,
                        byte_store=self.store,
                        id_key=self.id_key,
                        )
        self.splitter = TextSplitter()
        self.top_k = top_k


    @staticmethod
    def id_gen(self,doc: Document):
        return uuid.uuid5(multivec_retriver_conf['namespace'], doc.metadata['source']).hex

    @staticmethod
    def gen_doc_ids(self, docs: List[Document]):
        return [self.id_gen(doc) for doc in docs]

    def split_docs(self, docs: List[Document]):
        chunks = []
        for doc in docs:
            _id = self.id_gen(doc)
            _sub_docs = self.splitter.text_splitter.split_documents([doc])
            for _sub_doc in _sub_docs:
                _sub_doc.metadata[self.id_key] = _id
            chunks.extend(_sub_docs)
        return chunks

    def add_docs(self, docs: List[Document], asynchronous= True):
        chunks = self.split_docs(docs)
        doc_ids = self.gen_doc_ids(docs)
        if asynchronous:
            asyncio.run(self.client.aadd_docs(chunks))
        else:
            self.client.add_docs(chunks)
        self.retriever.docstore.mset(list(zip(doc_ids, docs)))

    def get_chunk(self, query: str, top_k=None, with_score=False):
        if top_k is None:
            top_k = self.top_k
        if with_score:
            return self.client.langchain_chroma.similarity_search_with_relevance_scores(query=query, k=top_k)
        else:
            return self.client.langchain_chroma.similarity_search(query=query, k=top_k)

    def aget_chunk(self, query: str, top_k=None, with_score=False):
        if top_k is None:
            top_k = self.top_k

        if with_score:
            return self.client.langchain_chroma.asimilarity_search_with_relevance_scores(query=query, k=top_k)
        else:
            return self.client.langchain_chroma.asimilarity_search(query=query, k=top_k)

    def get_doc(self, query: str):
        return self.client.langchain_chroma.similarity_search(query=query)

    def aget_doc(self, query: str):
        return self.retriever.aget_relevant_documents(query=query)
