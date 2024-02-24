from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.config import llm_conf
from langchain_core.prompts import ChatPromptTemplate

'''
Basic RAG v2 - refine, top_k

'''

llm_ = LLM()
llm = llm_.load_model()

retriever = Retriever(top_k=3)

template_question = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.

Always answer based only on the provided context. If the question can not be answered from the provided context, just say that you don't know, don't try to make up an answer.
<</SYS>>

Use the following pieces of context to answer the question at the end:


{context}


Question: {question}

Helpful Answer: [/INST]
"""

template_refine = '''
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.

Always answer based only on the provided context. If the question can not be answered from the provided context, just say that you don't know, don't try to make up an answer.
<</SYS>>

The original question is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer(only if needed) with some more context below.
------------
{context}
------------
Given the new context, refine the original answer to better answer the question. If you do update it. If the context isn't useful, return the original answer.

Helpful Answer: [/INST]
'''

prompt_template_question = ChatPromptTemplate.from_template(template_question)
prompt_template_refine = ChatPromptTemplate.from_template(template_refine)


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
