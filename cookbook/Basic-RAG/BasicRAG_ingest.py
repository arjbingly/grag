"""A cookbook demonstrating how to ingest pdf files for use with Basic RAG."""

from pathlib import Path

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient

# from grag.components.vectordb.chroma_client import ChromaClient

client = DeepLakeClient(collection_name="ci_test")
# client = ChromaClient(collection_name="ci_test")
retriever = Retriever(vectordb=client)

dir_path = Path(__file__).parents[2] / "data/test/pdfs/new_papers"
retriever.ingest(dir_path)

# retriever.aingest(dir_path)
