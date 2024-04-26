"""Class for Chroma vector database.

This module provides:

- ChromaClient
"""

from typing import List, Optional, Tuple, Union

import chromadb
from grag.components.embedding import Embedding
from grag.components.utils import configure_args
from grag.components.vectordb.base import VectorDB
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


@configure_args
class ChromaClient(VectorDB):
    """A class for connecting to a hosted Chroma Vectorstore collection.

    Attributes:
        host : str
            IP Address of hosted Chroma Vectorstore
        port : str
            port address of hosted Chroma Vectorstore
        collection_name : str
            name of the collection in the Chroma Vectorstore, each ChromaClient connects to a single collection
        embedding_type : str
            type of embedding used, supported 'sentence-transformers' and 'instructor-embedding'
        embedding_model : str
            model name of embedding used, should correspond to the embedding_type
        embedding_function
            a function of the embedding model, derived from the embedding_type and embedding_modelname
        client: chromadb.HttpClient
            Chroma API for client
        collection
            Chroma API for the collection
        langchain_client: langchain_community.vectorstores.Chroma
            LangChain wrapper for Chroma collection
    """

    def __init__(
            self,
            host: str,
            port: str,
            collection_name: str,
            embedding_type: str,
            embedding_model: str,
    ):
        """Initialize a ChromaClient object.

        Args:
        host: IP Address of hosted Chroma Vectorstore, defaults to argument from config file
        port: port address of hosted Chroma Vectorstore, defaults to argument from config file
        collection_name: name of the collection in the Chroma Vectorstore, defaults to argument from config file
        embedding_type: type of embedding used, supported 'sentence-transformers' and 'instructor-embedding', defaults to argument from config file
        embedding_model: model name of embedding used, should correspond to the embedding_type, defaults to argument from config file
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model

        self.embedding_function = Embedding(
            embedding_model=self.embedding_model, embedding_type=self.embedding_type
        ).embedding_function

        self.client = chromadb.HttpClient(host=self.host, port=self.port)  # type: ignore
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.langchain_client = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self.allowed_metadata_types = (str, int, float, bool)

    def __len__(self) -> int:
        """Count the number of chunks in the database."""
        return self.collection.count()

    def delete(self) -> None:
        """Delete all the chunks in the database collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        self.langchain_client = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def test_connection(self, verbose=True) -> int:
        """Tests connection with Chroma Vectorstore.

        Args:
            verbose: if True, prints connection status

        Returns:
            A random integer if connection is alive else None
        """
        response = self.client.heartbeat()
        if verbose:
            if response:
                print(f"Connection to {self.host}/{self.port} is alive..")
            else:
                print(f"Connection to {self.host}/{self.port} is not alive !!")
        return response

    def add_docs(self, docs: List[Document], verbose=True) -> None:
        """Adds documents to chroma vectorstore.

        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        """
        docs = self._filter_metadata(docs)
        for doc in (
                tqdm(docs, desc=f"Adding to {self.collection_name}:") if verbose else docs
        ):
            _id = self.langchain_client.add_documents([doc])

    async def aadd_docs(self, docs: List[Document], verbose=True) -> None:
        """Asynchronously adds documents to chroma vectorstore.

        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        """
        docs = self._filter_metadata(docs)
        if verbose:
            for doc in atqdm(
                    docs,
                    desc=f"Adding documents to {self.collection_name}",
                    total=len(docs),
            ):
                await self.langchain_client.aadd_documents([doc])
        else:
            for doc in docs:
                await self.langchain_client.aadd_documents([doc])

    def get_chunk(
            self, query: str, with_score: bool = False, top_k: Optional[int] = None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Returns the most similar chunks from the chroma database.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents

        """
        if with_score:
            return self.langchain_client.similarity_search_with_relevance_scores(
                query=query, k=top_k if top_k else 1
            )
        else:
            return self.langchain_client.similarity_search(
                query=query, k=top_k if top_k else 1
            )

    async def aget_chunk(
            self, query: str, with_score=False, top_k=None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Returns the most (cosine) similar chunks from the vector database, asynchronously.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents

        """
        if with_score:
            return await self.langchain_client.asimilarity_search_with_relevance_scores(
                query=query, k=top_k if top_k else 1
            )
        else:
            return await self.langchain_client.asimilarity_search(
                query=query, k=top_k if top_k else 1
            )
