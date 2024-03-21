from pathlib import Path
from typing import List, Union

from deeplake.core.vectorstore import VectorStore
from grag.components.embedding import Embedding
from grag.components.utils import get_config
from grag.components.vectordb.base import VectorDB
from langchain_community.vectorstores import DeepLake
from langchain_core.documents import Document
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

deeplake_conf = get_config()["deeplake"]


class DeepLakeClient(VectorDB):
    """A class for connecting to a DeepLake Vectorstore
    
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
        collection
            Chroma API for the collection
        langchain_client: langchain_community.vectorstores.DeepLake
            LangChain wrapper for DeepLake API
    """

    def __init__(self,
                 store_path: Union[str, Path],
                 embedding_model: str,
                 embedding_type: str,
                 ):
        self.store_path = Path(store_path)
        self.embedding_type: str = embedding_type
        self.embedding_model: str = embedding_model

        self.embedding_function = Embedding(
            embedding_model=self.embedding_model, embedding_type=self.embedding_type
        ).embedding_function

        self.client = VectorStore(path=self.store_path)
        self.langchain_client = DeepLake(path=self.store_path,
                                         embedding=self.embedding_function)
        self.allowed_metadata_types = (str, int, float, bool)

    def add_docs(self, docs: List[Document], verbose=True):
        """Adds documents to deeplake vectorstore
    
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
            _id = self.langchain_chroma.add_documents([doc])

    async def aadd_docs(self, docs: List[Document], verbose=True):
        """Asynchronously adds documents to chroma vectorstore
    
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
                await self.langchain_deeplake.aadd_documents([doc])
        else:
            for doc in docs:
                await self.langchain_deeplake.aadd_documents([doc])

    def get_chunk(self, query: str, with_score: bool = False, top_k: int = None):
        """Returns the most similar chunks from the deeplake database.

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

    async def aget_chunk(self, query: str, with_score=False, top_k=None):
        """Returns the most similar chunks from the deeplake database, asynchronously.

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
