"""Refine Chain
=======================
This cookbook demonstrates how to use the refine chain for BasicRAG.
For more information, refer to `RAG-PIPELINES <https://github.com/arjbingly/Capstone_5/blob/main/cookbook/Basic-RAG/RAG-PIPELINES.md
/>`_.

.. figure:: ../../_static/refine_chain_langchain_illustration.jpg
  :width: 800
  :alt: Refine Documents Chain Process
  :align: center

  Illustration of refine chain (Source: LangChain)

"""

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="test")
retriever = Retriever(vectordb=client)
rag = BasicRAG(doc_chain="refine")

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
