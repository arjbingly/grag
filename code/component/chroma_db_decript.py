import chromadb
import os
import pandas as pd

#%%
os.chdir('..') # goes to ../CapStone_5/code
os.chdir('..') # goes to ../CapStone_5/
print(os.getcwd())
#%%
VectorStorePath= "data/vectordb"
CollectionName = "gutenberg"
#%%
data_path = "data/Gutenberg/txt"
# client = chromadb.PersistentClient(path=VectorStorePath)
# collection = client.get_or_create_collection(name=CollectionName)
# #%%
# print(collection.peek())
# print(collection.count())
#%%
# filenames = os.listdir(data_path)
# filenames = [filename for filename in filenames if filename.endswith('.txt')]
# author_names = []
# titles = []
# for filename in filenames:
#     filename = filename.split('___')
#     author_name = filename[0]
#     title = filename[1].replace('.txt','')
#
#     author_names.append(author_name)
#     titles.append(title)
# df = pd.DataFrame({'filename':filenames, 'author':author_names, 'title':titles })

#%%
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#%%
def load_split_dir(data_path):
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader,
                                show_progress=True, use_multithreading=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    docs = loader.load_and_split(text_splitter)
    return docs
#%%
docs = load_split_dir(data_path)
print(f'Total Number of docs: {len(docs)}')
#%%
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

client = chromadb.PersistentClient(path=VectorStorePath)
collection = client.get_or_create_collection(name=CollectionName)

print(f'The {CollectionName} has {collection.count()} documenets')
print(f'A few are..')
print(collection.peek())

langchain_chroma = Chroma(
    client=client,
    collection_name=CollectionName,
    embedding_function=embedding_function,
)
db = langchain_chroma.from_documents(docs, embedding_function)
