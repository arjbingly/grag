import os
from pathlib import Path
import sys

sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from components.llm import LLM
from components.utils import process_llm_response
# from components.config import llm_conf


loader = DirectoryLoader("../components/new_papers", glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
# https://stackoverflow.com/a/77990923/13808323
embedding_instruction = 'Represent the document for retrival'
embedding_function = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl',
                                                   model_kwargs={"device": "cuda"})
embedding_function.embed_instruction = embedding_instruction

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding_function,)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm_ = LLM()
llm_model = llm_.load_model()

qa_chain = RetrievalQA.from_chain_type(llm=llm_model,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)

query = "What is Flash attention?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)
print("\n")
query = "How many tokens was LLaMA-2 trained on?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)
print("\n")
query = "How many examples do we need to provide for each tool?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)
print("\n")
query = "What are the best retrieval augmentations for LLMs?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)
