"""Custom Prompts
====================
This cookbook demonstrates how to use custom prompts with Basic RAG.


`Note that this cookbook assumes that you already have the 'Llama-2-13b-chat' LLM ready, `  
`for more details on how to quantize and fun an LLM locally, `
`refer to the LLM section under Getting Started.`

`Note that this cookbook also assumes that you have already ingested documents into a DeepLake collection called 'grag'`  
`for more details on how to ingest documents refer to the cookbook called` ``BasicRAG_ingest``.
"""

from grag.components.multivec_retriever import Retriever
from grag.components.prompt import Prompt
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

custom_prompt = Prompt(
    input_keys={"context", "question"},
    template="""Answer the following question based on the given context.
    question: {question}
    context: {context}
    answer: 
    """,
)

client = DeepLakeClient(collection_name="grag")
retriever = Retriever(vectordb=client)
rag = BasicRAG(model_name='Llama-2-13b-chat', custom_prompt=custom_prompt, retriever=retriever)

if __name__ == "__main__":
    while True:
        query = input("Query:")
        rag(query)
