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


`Note that this cookbook assumes that you already have the` ``Llama-2-13b-chat`` `LLM ready,`
`for more details on how to quantize and run an LLM locally,`
`refer to the LLM section under Getting Started.`
"""

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="grag")
retriever = Retriever(vectordb=client)

rag = BasicRAG(model_name="Llama-2-13b-chat", retriever=retriever)
# Note that doc_chain='stuff' is the default hence not passed to the class explicitly.


if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
