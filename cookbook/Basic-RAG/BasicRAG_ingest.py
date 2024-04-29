"""Document Ingestion
=======================
This cookbook demonstrates how to ingest pdf documents into a vector database.
"""

import asyncio
from pathlib import Path

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient

client = DeepLakeClient(collection_name="grag")

## Alternatively to use Chroma
# from grag.components.vectordb.chroma_client import ChromaClient
# client = ChromaClient(collection_name="grag")

ASYNC = True

retriever = Retriever(vectordb=client)

dir_path = Path(__file__).parents[2] / "data/pdf"  # path to the folder containing the pdfs

if ASYNC:
    asyncio.run(retriever.aingest(dir_path))
else:
    retriever.ingest(dir_path)
