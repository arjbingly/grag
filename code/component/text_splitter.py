from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from config import text_splitter_conf

import uuid
from typing import List


# %%
class TextSplitter:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=text_splitter_conf['chunk_size'],
                                                            chunk_overlap=text_splitter_conf['chunk_overlap'],
                                                            length_function=len,
                                                            is_separator_regex=False, )