from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from config import text_splitter_conf

import uuid
from typing import List


# %%
class TextSplitter:
    def __init__(self, id_key="doc_id"):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=text_splitter_conf['chunk_size'],
                                                            chunk_overlap=text_splitter_conf['chunk_overlap'],
                                                            length_function=len,
                                                            is_separator_regex=False, )
        self.id_key = id_key

    @staticmethod
    def id_gen(doc: Document):
        return uuid.uuid5(text_splitter_conf['namespace'], doc.metadata['source'])

    def split_docs(self, docs: List[Document]):
        doc_ids = [self.id_gen(doc) for doc in docs]
        split_docs = self.text_splitter.split_documents(docs)
        return split_docs, doc_ids

    def split_docs(self, docs: List[Document]):
        split_docs = []
        for doc in docs:
            _id = self.id_gen(doc)
            _sub_docs = self.text_splitter.split_documents([doc])
            for _sub_doc in sub_docs:
                _sub_doc.metadata[self.id_key] = _id
            split_docs.extend(_sub_docs)
        return split_docs