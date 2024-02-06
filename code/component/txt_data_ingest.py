import os
from tqdm import tqdm

import chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# %%
os.chdir('..')  # goes to ../CapStone_5/code
os.chdir('..')  # goes to ../CapStone_5/
print(os.getcwd())
# %%
## COFIGS
VectorStorePath = "data/vectordb"
CollectionName = "gutenberg"
EmbeddingModel = "all-mpnet-base-v2"

data_path = "data/Gutenberg/txt"


# %%
## FUNCS
def load_split_dir(data_path):
    loader = DirectoryLoader(data_path,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             show_progress=True,
                             use_multithreading=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100,
                                                   length_function=len,
                                                   is_separator_regex=False, )
    docs = loader.load_and_split(text_splitter)
    return docs

def add_docs(docs, client, collection_name, embedding_function):
    langchain_chroma = Chroma(client=client,
                              collection_name=CollectionName,
                              embedding_function=embedding_function, )

    for doc in tqdm(docs, desc='Adding Documents:'):
        _id = langchain_chroma.add_documents([doc])
# %%
docs = load_split_dir(data_path)
print(f'Total Number of docs: {len(docs)}')
# %%
embedding_function = SentenceTransformerEmbeddings(model_name=EmbeddingModel)
client = chromadb.PersistentClient(path=VectorStorePath)
collection = client.get_or_create_collection(name=CollectionName)

print('Before Adding Docs...')
print(f'The {CollectionName} has {collection.count()} documenets')
print(f'A few are..')
print(collection.peek())

add_docs(docs, client, CollectionName, embedding_function)
print('After Adding Docs...')
print(f'The {CollectionName} has {collection.count()} documents')
print(f'A few are..')
print(collection.peek())
