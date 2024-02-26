from pathlib import Path

from src.components.config import llm_conf
from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.utils import stuff_docs, load_prompt

# from prompts import
'''
Basic RAG v1 - stuff, chunks
    Given a query, retrieve similar chunks from vector database. Concat them into a single string, called context.
    Using the prompt template, call llm. Return chunk sources.
'''

llm_ = LLM()
llm = llm_.load_model()

retriever = Retriever(top_k=3)

prompt_name = 'Llama-2_QA_1.json'
prompt_path = Path(__file__).parent / 'prompts' / prompt_name
prompt_template = load_prompt(prompt_path)


def call_rag(query):
    retrieved_docs = retriever.get_chunk(query)
    context = stuff_docs(retrieved_docs)
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    sources = [doc.metadata["source"] for doc in retrieved_docs]
    return response, sources


if __name__ == "__main__":
    while True:
        query = input("Query:")
        response, sources = call_rag(query)
        if not llm_conf['std_out']:
            print(response)
        print(f'Sources: ')
        for index, source in enumerate(sources):
            print(f'\t{index}: {source}')
