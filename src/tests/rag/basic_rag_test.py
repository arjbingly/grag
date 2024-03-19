from typing import Text, List

from grag.rag.basic_rag import BasicRAG


def test_rag_stuff():
    rag = BasicRAG(doc_chain='stuff')
    response, sources = rag('What is simulated annealing?')
    assert isinstance(response, Text)
    assert isinstance(sources, List)
    assert all(isinstance(s, str) for s in sources)
    del rag.llm


def test_rag_refine():
    rag = BasicRAG(doc_chain='refine')
    response, sources = rag('What is simulated annealing?')
    # assert isinstance(response, Text)
    assert isinstance(response, List)
    assert all(isinstance(s, str) for s in response)
    assert isinstance(sources, List)
    assert all(isinstance(s, str) for s in sources)
    del rag.llm
