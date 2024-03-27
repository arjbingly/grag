from typing import List, Text

from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

client = DeepLakeClient(collection_name="test")
retriever = Retriever(vectordb=client)


def test_rag_stuff():
    rag = BasicRAG(doc_chain="stuff", retriever=retriever)
    response, sources = rag("What is Flash Attention?")
    assert isinstance(response, Text)
    assert isinstance(sources, List)
    assert all(isinstance(s, str) for s in sources)
    del rag.llm


def test_rag_refine():
    rag = BasicRAG(doc_chain="refine", retriever=retriever)
    response, sources = rag("What is Flash Attention?")
    assert isinstance(response, List)
    assert all(isinstance(s, str) for s in response)
    assert isinstance(sources, List)
    assert all(isinstance(s, str) for s in sources)
    del rag.llm
