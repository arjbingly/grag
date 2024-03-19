# add code folder to sys path
import os
from pathlib import Path

from grag.components.multivec_retriever import Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# %%%
# data_path = "data/pdf/9809"
data_path = Path(os.getcwd()).parents[1] / 'data' / 'Gutenberg' / 'txt'  # "data/Gutenberg/txt"
# %%
retriever = Retriever(top_k=3)

new_docs = True

if new_docs:
    # loading text files from data_path
    loader = DirectoryLoader(data_path,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             show_progress=True,
                             use_multithreading=True)
    print('Loading Files: ')
    docs = loader.load()
    # %%
    # limit docs for testing
    docs = docs[:100]

    # %%
    # adding chunks and parent doc
    retriever.add_docs(docs)
# %%
# testing retrival
query = 'Thomas H. Huxley'

# Retrieving the 3 most relevant small chunk
chunk_result = retriever.get_chunk(query)

# Retrieving the most relevant document
doc_result = retriever.get_doc(query)

# Ensuring that the length of chunk is smaller than length of doc
chunk_len = [len(chunk.page_content) for chunk in chunk_result]
print(f'Length of chunks : {chunk_len}')
doc_len = [len(doc.page_content) for doc in doc_result]
print(f'Length of doc    : {doc_len}')
len_test = [c_len < d_len for c_len, d_len in zip(chunk_len, doc_len)]
print(f'Is len of chunk less than len of doc?: {len_test} ')

# Ensuring both the chunk and document match the source
chunk_sources = [chunk.metadata['source'] for chunk in chunk_result]
doc_sources = [doc.metadata['source'] for doc in doc_result]
source_test = [source[0] == source[1] for source in zip(chunk_sources, doc_sources)]
print(f'Does source of chunk and doc match? : {source_test}')

# # Ensuring both the chunk and document match the source
# source_test = chunk_result.metadata['source'] == doc_result.metadata['source']
# print(f'Does source of chunk and doc match? : {source_test}')
# # Ensuring that the length of chunk is smaller than length of doc
# len_test = chunk_len <= doc_len
# print(f'Is len of chunk less than len of doc?: {len_test} ')
# %%
