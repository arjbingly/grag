from typing import List

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from tqdm import tqdm

from .embedding import Embedding
from .utils import get_config

chroma_conf = get_config()['chroma']


class ChromaClient:
    """
    A class for connecting to a hosted Chroma Vectorstore collection.

    Attributes:
        host : str
            IP Address of hosted Chroma Vectorstore
        port : str
            port address of hosted Chroma Vectorstore
        collection_name : str
            name of the collection in the Chroma Vectorstore, each ChromaClient connects to a single collection
        embedding_type : str
            type of embedding used, supported 'sentence-transformers' and 'instructor-embedding'
        embedding_modelname : str
            model name of embedding used, should correspond to the embedding_type
        embedding_function
            a function of the embedding model, derived from the embedding_type and embedding_modelname
        chroma_client
            Chroma API for client
        collection
            Chroma API for the collection
        langchain_chroma
            LangChain wrapper for Chroma collection
    """

    def __init__(self,
                 host=chroma_conf['host'],
                 port=chroma_conf['port'],
                 collection_name=chroma_conf['collection_name'],
                 embedding_type=chroma_conf['embedding_type'],
                 embedding_model=chroma_conf['embedding_model']):
        """
        Args:
            host: IP Address of hosted Chroma Vectorstore, defaults to argument from config file
            port: port address of hosted Chroma Vectorstore, defaults to argument from config file
            collection_name: name of the collection in the Chroma Vectorstore, defaults to argument from config file
            embedding_type: type of embedding used, supported 'sentence-transformers' and 'instructor-embedding', defaults to argument from config file
            embedding_model: model name of embedding used, should correspond to the embedding_type, defaults to argument from config file
        """

        self.host: str = host
        self.port: str = port
        self.collection_name: str = collection_name
        self.embedding_type: str = embedding_type
        self.embedding_model: str = embedding_model

        self.embedding_function = Embedding(embedding_model=self.embedding_model,
                                            embedding_type=self.embedding_type).embedding_function

        self.chroma_client = chromadb.HttpClient(host=self.host, port=self.port)
        self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
        self.langchain_chroma = Chroma(client=self.chroma_client,
                                       collection_name=self.collection_name,
                                       embedding_function=self.embedding_function, )
        self.allowed_metadata_types = (str, int, float, bool)

    def test_connection(self, verbose=True):
        '''
        Tests connection with Chroma Vectorstore

        Args:
            verbose: if True, prints connection status

        Returns:
            A random integer if connection is alive else None
        '''
        response = self.chroma_client.heartbeat()
        if verbose:
            if response:
                print(f'Connection to {self.host}/{self.port} is alive..')
            else:
                print(f'Connection to {self.host}/{self.port} is not alive !!')
        return response

    async def aadd_docs(self, docs: List[Document], verbose=True):
        '''
        Asynchronously adds documents to chroma vectorstore

        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        '''
        docs = self._filter_metadata(docs)
        # tasks = [await self.langchain_chroma.aadd_documents([doc]) for doc in docs]
        # if verbose:
        #     await tqdm_asyncio.gather(*tasks, desc=f'Adding to {self.collection_name}')
        # else:
        #     await asyncio.gather(*tasks)
        await self.langchain_chroma.aadd_documents(docs, verbose=verbose)

    def add_docs(self, docs: List[Document], verbose=True):
        '''
        Adds documents to chroma vectorstore

         Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        '''
        docs = self._filter_metadata(docs)
        for doc in (tqdm(docs, desc=f'Adding to {self.collection_name}:') if verbose else docs):
            _id = self.langchain_chroma.add_documents([doc])

    def _filter_metadata(self, docs: List[Document]):
        return filter_complex_metadata(docs, allowed_types=self.allowed_metadata_types)
