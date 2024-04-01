"""A cookbook demonstrating how to ingest pdf files for use with Basic RAG."""

from pathlib import Path

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient

client = DeepLakeClient(collection_name="ci_test")
retriever = Retriever(vectordb=client)

dir_path = Path(__file__).parents[2] / "data/ci_test/"

retriever.ingest(dir_path)
