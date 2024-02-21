from components.llm import LLM
from components.multivec_retriever import Retriever
from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub


from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
# from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.schema import format_document

retriever = Retriever(top_k=3)
retriever_lc = retriever.vectorstore_retriever
# retriever_lc = retriever.retriever

llm_ = LLM()
llm_model = llm_.load_model()

# query = 'Explain Non-Deterministic Quantum Communication Complexity.'
query = 'What types of dependencies does dependence analysis identify in loop programs?'

# llm_model.invoke()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm_model, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever_lc, combine_docs_chain)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retrieval_chain, "question": RunnablePassthrough()}
)
chain = (
    {'input': RunnablePassthrough()} |
    setup_and_retrieval |
    prompt |
    llm_model |
    output_parser
)

# from langchain.schema import format_document
# from langchain.prompts.prompt import PromptTemplate
# from operator import itemgetter
#
# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
#
#
# def combine_docs(docs) -> str:
#     doc_prompt = PromptTemplate.from_template(template="{page_content}")
#     return "\n\n".join(
#         format_document(doc, doc_prompt) for doc in docs)
#
#
# # retrieved_docs = RunnablePassthrough.assign(
# #     docs=retriever_lc
# # )
# retrive_docs = RunnableParallel(
#     {"docs": retriever_lc}
# )
# setup_and_retrieval = RunnableParallel(
#     {"context": itemgetter('docs') | combine_docs, "question": RunnablePassthrough()}
# )
#
# # answer = {
# #     'answer': {'input': RunnablePassthrough()} |
# #               setup_and_retrieval |
# #               prompt |
# #               llm_model |
# #               output_parser,
# #     # 'docs': itemgetter('docs')
# # }
#
# chain = (retrive_docs |
#          setup_and_retrieval |
#          prompt |
#          llm_model |
#          output_parser
#          )

chain.invoke(query)

# def invoke_chain(query: str):
#     retrieval_chain.invoke({"input": query})
#
#
# # invoke_chain(query)
