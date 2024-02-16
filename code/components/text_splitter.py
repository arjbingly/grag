from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import text_splitter_conf


# %%
class TextSplitter:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=text_splitter_conf['chunk_size'],
                                                            chunk_overlap=text_splitter_conf['chunk_overlap'],
                                                            length_function=len,
                                                            is_separator_regex=False, )