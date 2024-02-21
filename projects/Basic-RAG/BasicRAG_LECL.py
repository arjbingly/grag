from components.multivec_retriever import Retriever
from components.llm import LLM

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

retriever = Retriever()
llm_ = LLM()
llm = llm_.load_model()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": lambda x: retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)
# question = 'Explain Non-Deterministic Quantum Communication Complexity.'
question = 'What types of dependencies does dependence analysis identify in loop programs?'

response = chain.invoke(question)
print(response)
