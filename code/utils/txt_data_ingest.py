import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from pathlib import Path
import sys
sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from components.text_splitter import TextSplitter
# %%
## FUNCS
def load_split_dir(data_path):
    loader = DirectoryLoader(data_path,
                             glob="*.txt",
                             loader_cls=TextLoader,
                             show_progress=True,
                             use_multithreading=True)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=text_splitter_conf['chunk_size'],
    #                                                chunk_overlap=text_splitter_conf['chunk_overlap'],
    #                                                length_function=len,
    #                                                is_separator_regex=False, )
    text_splitter = TextSplitter().text_splitter
    docs = loader.load_and_split(text_splitter)
    return docs
