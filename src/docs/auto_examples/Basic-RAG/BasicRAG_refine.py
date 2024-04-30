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


`Note that this cookbook assumes that you already have` ``Llama-2-13b-chat`` `LLM ready,`
`for more details on how to quantize and run an LLM locally,`
`refer to the LLM section under Getting Started.`
"""

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="grag")
retriever = Retriever(vectordb=client)
rag = BasicRAG(model_name="Llama-2-13b-chat", doc_chain="refine", retriever=retriever)

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
