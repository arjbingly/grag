import asyncio
import uuid
from pathlib import Path
from typing import List, Union

from grag.components.chroma_client import ChromaClient
from grag.components.parse_pdf import ParsePDF
from grag.components.text_splitter import TextSplitter
from grag.components.utils import get_config
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

multivec_retriever_conf = get_config()['multivec_retriever']


class Retriever:
    """
    A class for multi vector retriever, it connects to a vector database and a local file store.
    It is used to return most similar chunks from a vector store but has the additional functionality
    to return a linked document, chunk, etc.

    Attributes:
        store_path: Path to the local file store
        id_key: A key prefix for identifying documents
        client: ChromaClient class instance from components.chroma_client
        store: langchain.storage.LocalFileStore object, stores the key value pairs of document id and parent file
        retriever: langchain.retrievers.multi_vector.MultiVectorRetriever class instance, 
                    langchain's multi-vector retriever
        splitter: TextSplitter class instance from components.text_splitter
        namespace: Namespace for producing unique id
        top_k: Number of top chunks to return from similarity search.

    """

    def __init__(self,
                 store_path: str = multivec_retriever_conf['store_path'],
                 id_key: str = multivec_retriever_conf['id_key'],
                 namespace: str = multivec_retriever_conf['namespace'],
                 top_k=1):
        """
        Args:
            store_path: Path to the local file store, defaults to argument from config file
            id_key: A key prefix for identifying documents, defaults to argument from config file
            namespace: A namespace for producing unique id, defaults to argument from config file
            top_k: Number of top chunks to return from similarity search, defaults to 1
        """
        self.store_path = store_path
        self.id_key = id_key
        self.namespace = uuid.UUID(namespace)
        self.client = ChromaClient()
        self.store = LocalFileStore(self.store_path)
        self.retriever = MultiVectorRetriever(
            vectorstore=self.client.langchain_chroma,
            byte_store=self.store,
            id_key=self.id_key,
        )
        self.splitter = TextSplitter()
        self.top_k: int = top_k
        self.retriever.search_kwargs = {'k': self.top_k}

    def id_gen(self, doc: Document) -> str:
        """
        Takes a document and returns a unique id (uuid5) using the namespace and document source.
        This ensures that a  single document always gets the same unique id.

        Args:
            doc: langchain_core.documents.Document

        Returns:
            string of hexadecimal uuid
        """
        return uuid.uuid5(self.namespace, doc.metadata['source']).hex

    def gen_doc_ids(self, docs: List[Document]) -> List[str]:
        """
        Takes a list of documents and produces a list of unique id, refer id_gen method for more details.

        Args:
            docs: List of langchain_core.documents.Document

        Returns:
            List of hexadecimal uuid

        """
        return [self.id_gen(doc) for doc in docs]

    def split_docs(self, docs: List[Document]) -> List[Document]:
        """
        Takes a list of documents and splits them into smaller chunks using TextSplitter from components.text_splitter
        Also adds the unique parent document id into metadata

        Args:
            docs: List of langchain_core.documents.Document

        Returns:
            list of chunks after splitting

        """
        chunks = []
        for doc in docs:
            _id = self.id_gen(doc)
            _sub_docs = self.splitter.text_splitter.split_documents([doc])
            for _sub_doc in _sub_docs:
                _sub_doc.metadata[self.id_key] = _id
            chunks.extend(_sub_docs)
        return chunks

    def add_docs(self, docs: List[Document]):
        """
        Takes a list of documents, splits them using the split_docs method and then adds them into the vector database
        and adds the parent document into the file store.
        Args:
            docs: List of langchain_core.documents.Document

        Returns:
            None

        """
        chunks = self.split_docs(docs)
        doc_ids = self.gen_doc_ids(docs)
        self.client.add_docs(chunks)
        self.retriever.docstore.mset(list(zip(doc_ids, docs)))

    async def aadd_docs(self, docs: List[Document]):
        """
          Takes a list of documents, splits them using the split_docs method and then adds them into the vector database
          and adds the parent document into the file store.
          Args:
              docs: List of langchain_core.documents.Document

          Returns:
              None

          """
        chunks = self.split_docs(docs)
        doc_ids = self.gen_doc_ids(docs)
        await asyncio.run(self.client.aadd_docs(chunks))
        self.retriever.docstore.mset(list(zip(doc_ids)))

    def get_chunk(self, query: str, with_score=False, top_k=None):
        """
        Returns the most (cosine) similar chunks from the vector database.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents

        """
        if with_score:

            return self.client.langchain_chroma.similarity_search_with_relevance_scores(
                query=query,
                **{'k': top_k} if top_k else self.retriever.search_kwargs
            )
        else:
            return self.client.langchain_chroma.similarity_search(
                query=query,
                **{'k': top_k} if top_k else self.retriever.search_kwargs
            )

    async def aget_chunk(self, query: str, with_score=False, top_k=None):
        """
        Returns the most (cosine) similar chunks from the vector database, asynchronously.

        Args:
            query: A query string
            with_score: Outputs scores of returned chunks
            top_k: Number of top similar chunks to return, if None defaults to self.top_k

        Returns:
            list of Documents

        """
        if with_score:
            return await self.client.langchain_chroma.asimilarity_search_with_relevance_scores(
                query=query,
                **{'k': top_k} if top_k else self.retriever.search_kwargs
            )
        else:
            return await self.client.langchain_chroma.asimilarity_search(
                query=query,
                **{'k': top_k} if top_k else self.retriever.search_kwargs
            )

    def get_doc(self, query: str):
        """
        Returns the parent document of the most (cosine) similar chunk from the vector database.

        Args:
            query: A query string
        Returns:
             Documents

        """
        return self.retriever.get_relevant_documents(query=query)

    async def aget_doc(self, query: str):
        """
        Returns the parent documents of the most (cosine) similar chunks from the vector database.
        Args:
            query: A query string
        Returns:
             Documents

        """
        return await self.retriever.aget_relevant_documents(query=query)

    def get_docs_from_chunks(self, chunks: List[Document], one_to_one=False):
        """
        Returns the parent documents of chunks.
        Args:
            chunks: chunks from vector store
            one_to_one: if True, returns parent doc for each chunk
        Returns:
             parent documents
        """
        ids = []
        for d in chunks:
            if one_to_one:
                if self.id_key in d.metadata:
                    ids.append(d.metadata[self.id_key])
                docs = self.retriever.docstore.mget(ids)
                return docs
            else:
                if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                    ids.append(d.metadata[self.id_key])
                docs = self.retriever.docstore.mget(ids)
                return [d for d in docs if d is not None]

    def ingest(self,
               dir_path: Union[str, Path],
               glob_pattern: str = "**/*.pdf",
               dry_run: bool = False,
               verbose: bool = True,
               parser_kwargs: dict = None):
        """
        Ingests the files in directory
        Args:
            dir_path: path to the directory
            glob_pattern: glob pattern to identify files
            dry_run: if True, does not ingest any files
            verbose: if True, shows progress
            parser_kwargs: arguments to pass to the parser

        """
        _formats_to_add = ['Text', 'Tables']
        filepath_gen = Path(dir_path).glob(glob_pattern)
        if parser_kwargs:
            parser = ParsePDF(parser_kwargs)
        else:
            parser = ParsePDF()
        if verbose:
            num_files = len(list(Path(dir_path).glob(glob_pattern)))
            pbar = tqdm(filepath_gen, total=num_files, desc='Ingesting Files')
            for filepath in pbar:
                if not dry_run:
                    pbar.set_postfix_str(f'Parsing file - {filepath.relative_to(dir_path)}')
                    docs = parser.load_file(filepath)
                    pbar.set_postfix_str(f'Adding file - {filepath.relative_to(dir_path)}')
                    for format_key in _formats_to_add:
                        self.add_docs(docs[format_key])
                    print(f'Completed adding - {filepath.relative_to(dir_path)}')
                else:
                    print(f'DRY RUN: found - {filepath.relative_to(dir_path)}')

    async def aingest(self,
                      dir_path: Union[str, Path],
                      glob_pattern: str = "**/*.pdf",
                      dry_run: bool = False,
                      verbose: bool = True,
                      parser_kwargs: dict = None):
        """
        Asynchronously ingests the files in directory
        Args:
            dir_path: path to the directory
            glob_pattern: glob pattern to identify files
            dry_run: if True, does not ingest any files
            verbose: if True, shows progress
            parser_kwargs: arguments to pass to the parser

        """
        _formats_to_add = ['Text', 'Tables']
        filepath_gen = Path(dir_path).glob(glob_pattern)
        if parser_kwargs:
            parser = ParsePDF(parser_kwargs)
        else:
            parser = ParsePDF()
        if verbose:
            num_files = len(list(Path(dir_path).glob(glob_pattern)))
            pbar = atqdm(filepath_gen, total=num_files, desc='Ingesting Files')
            for filepath in pbar:
                if not dry_run:
                    pbar.set_postfix_str(f'Parsing file - {filepath.relative_to(dir_path)}')
                    docs = parser.load_file(filepath)
                    pbar.set_postfix_str(f'Adding file - {filepath.relative_to(dir_path)}')
                    for format_key in _formats_to_add:
                        await self.aadd_docs(docs[format_key])
                    print(f'Completed adding - {filepath.relative_to(dir_path)}')
                else:
                    print(f'DRY RUN: found - {filepath.relative_to(dir_path)}')
