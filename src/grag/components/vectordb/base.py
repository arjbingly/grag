"""Abstract base class for vector database clients.

This module provides:

- VectorDB
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document


class VectorDB(ABC):
    """Abstract base class for vector database clients."""

    def __init__(self):
        """Initialize the vector."""
        self.allowed_metadata_types = ()

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks in the vector database."""
        ...

    @abstractmethod
    def delete(self) -> None:
        """Delete all chunks in the vector database."""

    @abstractmethod
    def add_docs(self, docs: List[Document], verbose: bool = True) -> None:
        """Adds documents to the vector database.

        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        """
        ...

    @abstractmethod
    async def aadd_docs(self, docs: List[Document], verbose: bool = True) -> None:
        """Adds documents to the vector database (asynchronous).

        Args:
            docs: List of Documents
            verbose: Show progress bar

        Returns:
            None
        """
        ...

    @abstractmethod
    def get_chunk(
        self, query: str, with_score: bool = False, top_k: Optional[int] = None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
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
    async def aget_chunk(
        self, query: str, with_score: bool = False, top_k: Optional[int] = None
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Returns the most similar chunks from the vector database (asynchronous).

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents
        """
        ...

    def _filter_metadata(self, docs: List[Document]) -> List[Document]:
        return filter_complex_metadata(docs, allowed_types=self.allowed_metadata_types)
