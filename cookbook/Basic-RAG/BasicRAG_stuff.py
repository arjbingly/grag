"""A cookbook demonstrating how to use Basic RAG with stuff chain using DeepLake as client."""

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="test")
retriever = Retriever(vectordb=client)

rag = BasicRAG(doc_chain="stuff", retriever=retriever)

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
