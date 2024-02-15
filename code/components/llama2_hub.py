# from dotenv import load_dotenv
# import os
#
# load_dotenv()
# # Access the environment variable
# HF_TOKEN = os.getenv("AUTH_TOKEN")
# from huggingface_hub import login
#
# login(token=HF_TOKEN)
from getpass import getpass
HUGGINGFACEHUB_API_TOKEN = getpass()
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

import torch
print("CUDA: ", torch.cuda.is_available())
import textwrap

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

loader = DirectoryLoader("./new_papers", glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding_function,
                                 persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm = HuggingFaceHub(
    # repo_id="meta-llama/Llama-2-7b-chat-hf",
    # repo_id="TheBloke/Llama-2-13B-chat-GGUF",
    task="text-generation",
    # huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


#%%
query = "What is Flash attention?"
llm_response = qa_chain(query)
# print(llm_response)
process_llm_response(llm_response)
