from typing import List, Text

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="ci_test", read_only=True)
retriever = Retriever(vectordb=client)


def test_rag_stuff():
    rag = BasicRAG(doc_chain="stuff", retriever=retriever,
                   llm_kwargs={"model_name": "Llama-2-7b-chat", "n_gpu_layers": "-1"})
    response, retrieved_docs = rag("What is Flash Attention?")
    sources = [doc.metadata["source"] for doc in retrieved_docs]
    assert isinstance(response, Text)
    assert isinstance(sources, List)
    assert all(isinstance(s, str) for s in sources)
    del rag.llm


def test_rag_refine():
    rag = BasicRAG(doc_chain="refine", retriever=retriever,
                   llm_kwargs={"model_name": "Llama-2-7b-chat", "n_gpu_layers": "-1"})
    response, retrieved_docs = rag("What is Flash Attention?")
    sources = [doc.metadata["source"] for doc in retrieved_docs]
    assert isinstance(response, List)
    assert all(isinstance(s, str) for s in response)
    assert isinstance(sources, List)
    assert all(isinstance(s, str) for s in sources)
    del rag.llm
