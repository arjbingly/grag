"""Custom Prompts
====================
This cookbook demonstrates how to use custom prompts with Basic RAG.
"""

from grag.components.prompt import Prompt
from grag.rag.basic_rag import BasicRAG

custom_prompt = Prompt(
    input_keys={"context", "question"},
    template="""Answer the following question based on the given context.
    question: {question}
    context: {context}
    answer: 
    """,
)
rag = BasicRAG(doc_chain="stuff", custom_prompt=custom_prompt)
