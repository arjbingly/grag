"""A cookbook demonstrating how to ingest pdf files for use with Basic RAG."""

import asyncio
from pathlib import Path

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient

# from grag.components.vectordb.chroma_client import ChromaClient

# Initialize the client by giving a collection name - DeepLakeClient or ChromaClient
client = DeepLakeClient(collection_name="arxiv")
# client = ChromaClient(collection_name="arxiv")

# Initialize the retriever by passing in the client 
# Note that: if no client is passed the retriever class will take config from the config file.
retriever = Retriever(vectordb=client)

# The path to the folder with pdfs
dir_path = Path(__file__).parents[2] / "data/pdf"

# Either
#   1. To run synchronously (slow)
# retriever.ingest(dir_path)
#   2. To run asynchronously
asyncio.run(retriever.aingest(dir_path))
