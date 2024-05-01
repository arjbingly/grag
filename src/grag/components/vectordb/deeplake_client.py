"""Class for DeepLake vector database.

This module provides:

— DeepLakeClient
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from grag.components.embedding import Embedding
from grag.components.utils import configure_args
from grag.components.vectordb.base import VectorDB
from langchain_community.vectorstores import DeepLake
from langchain_core.documents import Document
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


@configure_args
class DeepLakeClient(VectorDB):
    """A class for connecting to a DeepLake Vectorstore.

    Attributes:
        store_path : str, Path
            The path to store the DeepLake vectorstore.
        embedding_type : str
            type of embedding used, supported 'sentence-transformers' and 'instructor-embedding'
        embedding_model : str
            model name of embedding used, should correspond to the embedding_type
        embedding_function
            a function of the embedding model, derived from the embedding_type and embedding_modelname
        client: deeplake.core.vectorstore.VectorStore
            DeepLake API
        collection_name: str
             The name of the collection where the vectors are stored.
        langchain_client: langchain_community.vectorstores.DeepLake
            LangChain wrapper for DeepLake API.
    """

    def __init__(
        self,
        store_path: Union[str, Path] = Path("data/vectordb"),
        collection_name: str = "grag",
        embedding_type: str = "instructor-embedding",
        embedding_model: str = "kunlp/instructor-xl",
        read_only: bool = False,
    ):
        """Initialize a DeepLakeClient object.

        Args:
            store_path: path to the deeplake vectorstore, defaults to 'data/vectordb'
            collection_name: name of the collection in the DeepLake Vectorstore, defaults to 'grag'
            embedding_type: type of embedding used, supported 'sentence-transformers' and 'instructor-embedding',
                            defaults to instructor-embedding
            embedding_model: model name of embedding used, should correspond to the embedding_type,
                             defaults to hkunlp/instructor-xl
            read_only: flag indicating whether the client is read-only, defaults to False.
        """
        self.store_path = Path(store_path)
        self.collection_name = collection_name
        self.read_only = read_only
        self.embedding_type: str = embedding_type
        self.embedding_model: str = embedding_model

        self.embedding_function = Embedding(
            embedding_model=self.embedding_model, embedding_type=self.embedding_type
        )

        # self.client = VectorStore(path=self.store_path / self.collection_name)
        self.langchain_client = DeepLake(
            dataset_path=str(self.store_path / self.collection_name),
            embedding=self.embedding_function,
            read_only=self.read_only,
        )
        self.client = self.langchain_client.vectorstore
        self.allowed_metadata_types = (str, int, float, bool)

    def __str__(self):
        str_string = "DeepLakeClient("
        str_string += f"\tstore_path: {self.store_path},\n"
        str_string += f"\tcollection_name: {self.collection_name},\n"
        str_string += f"\tread_only: {self.read_only},\n"
        str_string += f"\tembedding :{self.embedding_function}\n"
        str_string += ")"
        return str_string

    def __len__(self) -> int:
        """Number of chunks in the vector database."""
        return self.client.__len__()

    def delete(self) -> None:
        """Delete all chunks in the vector database."""
        self.client.delete(delete_all=True)

    def add_docs(self, docs: List[Document], verbose=True) -> None:
        """Adds documents to deeplake vectorstore.

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
        """Returns the most similar chunks from the deeplake database.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents

        """
        if with_score:
            return self.langchain_client.similarity_search_with_score(
                query=query, k=top_k if top_k else 1
            )
        else:
            return self.langchain_client.similarity_search(
                query=query, k=top_k if top_k else 1
            )

    async def aget_chunk(
        self, query: str, with_score=False, top_k=None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Returns the most similar chunks from the deeplake database, asynchronously.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents

        """
        if with_score:
            return await self.langchain_client.asimilarity_search_with_score(
                query=query, k=top_k if top_k else 1
            )
        else:
            return await self.langchain_client.asimilarity_search(
                query=query, k=top_k if top_k else 1
            )
