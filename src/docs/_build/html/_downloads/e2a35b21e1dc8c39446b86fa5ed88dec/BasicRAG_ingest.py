"""Document Ingestion
=======================
This cookbook demonstrates how to ingest documents into a vector database.
"""

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient

client = DeepLakeClient(collection_name="your_collection_name")

## Alternatively to use Chroma
# from grag.components.vectordb.chroma_client import ChromaClient
# client = ChromaClient(collection_name="ci_test")

retriever = Retriever(vectordb=client)

dir_path = "data/pdf"  # path to pdf files
retriever.ingest(dir_path)
