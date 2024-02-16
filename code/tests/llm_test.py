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
from components.config import llm_conf

# load docs from path
path = "../../data/new_papers"
loader = DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader)
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
                                 embedding=embedding_function)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

llm_ = LLM()

models_to_test = ['Llama-2-7b-chat',
                  'Llama-2-13b-chat',
                  'Mixtral-8x7B-Instruct-v0.1']

pipeline_list = ['llama_cpp', 'hf']

def test_model(model_list, pipeline_list):
    for pipeline in pipeline_list:
        print(f'****** TESTING PIPELINE: {pipeline} ******')
        for model_name in model_list:
            print(f'***** MODEL: {model_name} *****')
            model = llm_.load_model(model_name=model_name,
                                    pipeline=pipeline)
            qa_chain = RetrievalQA.from_chain_type(llm=model,
                                                   chain_type="stuff",
                                                   retriever=retriever,
                                                   return_source_documents=True)
            queries = ["What is Flash attention?",
                       "How many tokens was LLaMA-2 trained on?",
                       "How many examples do we need to provide for each tool?",
                       "What are the best retrieval augmentations for LLMs?"
                       ]
            for query in queries:
                print(f'Query: {query}')
                llm_response = qa_chain(query)
                process_llm_response(llm_response)
                print("\n")

            del model, qa_chain


if __name__=="__main__":
    test_model(models_to_test, pipeline_list)
