from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

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
