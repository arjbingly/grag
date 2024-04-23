"""Stuff Chain
=======================
This cookbook demonstrates how to use the stuff chain for BasicRAG.
For more information, refer to `RAG-PIPELINES <https://github.com/arjbingly/Capstone_5/blob/main/cookbook/Basic-RAG/RAG-PIPELINES.md
/>`_.

.. figure:: ../../_static/stuff_chain_langchain_illustration.jpg
  :width: 800
  :alt: Stuff Documents Chain Process
  :align: center

  Illustration of stuff chain (Source: LangChain)

"""

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
