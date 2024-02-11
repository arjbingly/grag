from multivec_retriever import Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
#%%%
os.chdir('..')
os.chdir('..')
data_path = "data/Gutenberg/txt"
#%%
# init multi vector retriever
retriever = Retriever()
# loading text files from data_path
loader = DirectoryLoader(data_path,
                         glob="*.txt",
                         loader_cls=TextLoader,
                         show_progress=True,
                         use_multithreading=True)
print('Loading Files: ')
docs = loader.load()
#%%
# limit docs for testing
docs = docs[:100]
#%%
# adding chunks and parent doc
retriever.add_docs(docs)
#%%
# testing retrival
query = 'Thomas H. Huxley'
# Retrieving the most relevant small chunk
chunk_result = retriever.get_chunk(query)[0]
chunk_len = len(chunk_result.page_content)
print(f'Length of chunk : {chunk_len}')
# Retrieving the most relevant document
doc_result = retriever.get_doc(query)[0]
doc_len = len(doc_result.page_content)
print(f'Length of doc   : {doc_len}')
# Ensuring both the chunk and document match the source
source_test = chunk_result.metadata['source'] == doc_result.metadata['source']
print(f'Does source of chunk and doc match? : {source_test}')
# Ensuring that the length of chunk is smaller than length of doc
len_test = chunk_len <= doc_len
print(f'Is len of chunk less than len of doc?: {len_test} ')
#%%

