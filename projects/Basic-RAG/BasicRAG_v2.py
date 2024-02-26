from pathlib import Path

from src.components.config import llm_conf
from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.utils import load_prompt

'''
Basic RAG v2 - refine, top_k

'''

llm_ = LLM()
llm = llm_.load_model()

retriever = Retriever(top_k=3)

prompts_path = Path(__file__).parent / 'prompts'
prompt_name_question = 'Llama-2_QA_1.json'
prompt_template_question = load_prompt(prompts_path / prompt_name_question)
prompt_name_refine = 'Llama-2_QA-refine_1.json'
prompt_template_refine = load_prompt(prompts_path / prompt_name_refine)


def call_rag(query):
    retrieved_docs = retriever.get_chunk(query)
    sources = [doc.metadata["source"] for doc in retrieved_docs]
    responses = []
    for index, doc in enumerate(retrieved_docs):
        if index == 0:
            prompt = prompt_template_question.format(context=doc.page_content,
                                                     question=query)
            response = llm.invoke(prompt)
            responses.append(response)
        else:
            prompt = prompt_template_refine.format(context=doc.page_content,
                                                   question=query,
                                                   existing_answer=responses[-1])
            response = llm.invoke(prompt)
            responses.append(response)
    return responses, sources


if __name__ == "__main__":
    while True:
        query = input("Query:")
        responses, sources = call_rag(query)
        if not llm_conf['std_out']:
            print(responses[-1])
        print(f'Sources: ')
        for index, source in enumerate(sources):
            print(f'\t{index}: {source}')
