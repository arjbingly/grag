from abc import ABC, abstractmethod
from typing import List

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document


class VectorDB(ABC):
    @abstractmethod
    def add_docs(self, docs: List[Document], verbose: bool = True):
        """Adds documents to the vector database.
        
        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        """
        ...

    @abstractmethod
    async def aadd_docs(self, docs: List[Document], verbose: bool = True):
        """Adds documents to the vector database (asynchronous).

        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        """
        ...

    @abstractmethod
    def get_chunk(self, query: str, with_score: bool = False, top_k: int = None):
        """Returns the most similar chunks from the vector database.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents
        """
        ...

    @abstractmethod
    async def aget_chunk(self, query: str, with_score: bool = False, top_k: int = None):
        """Returns the most similar chunks from the vector database. (asynchronous)

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents
        """
        ...

    def _filter_metadata(self, docs: List[Document]):
        return filter_complex_metadata(docs, allowed_types=self.allowed_metadata_types)
