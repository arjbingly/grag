"""A cookbook demonstrating how to ingest pdf files for use with BasicRAG."""

from pathlib import Path

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient

# from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="test")
retriever = Retriever(vectordb=client)

dir_path = Path(__file__).parent / "some_dir"

retriever.ingest(dir_path)
# rag = BasicRAG(doc_chain="refine")
