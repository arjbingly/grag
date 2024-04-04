import os
import shutil
from pathlib import Path

from grag.components.multivec_retriever import Retriever
from grag.components.utils import get_config
from grag.components.vectordb.deeplake_client import DeepLakeClient
from langchain_core.documents import Document

config = get_config()

test_path = Path(config['data']['data_path']) / 'vectordb/test_retriever'
if os.path.exists(test_path):
    shutil.rmtree(test_path)
    print('Deleting test retriever: {}'.format(test_path))

# client = DeepLakeClient(collection_name="test_retriever")
# retriever = Retriever(vectordb=client)  # pass test collection

doc = Document(page_content="Hello worlds", metadata={"source": "bars"})


def test_retriever_id_gen():
    client = DeepLakeClient(collection_name="test_retriever")
    retriever = Retriever(vectordb=client)
    doc = Document(page_content="Hello world", metadata={"source": "bar"})
    id_ = retriever.id_gen(doc)
    assert isinstance(id_, str)
    assert len(id_) == 32
    doc.page_content = doc.page_content + 'ABC'
    id_1 = retriever.id_gen(doc)
    assert id_ == id_1
    doc.metadata["source"] = "bars"
    id_1 = retriever.id_gen(doc)
    assert id_ != id_1
    del client, retriever


def test_retriever_gen_doc_ids():
    client = DeepLakeClient(collection_name="test_retriever")
    retriever = Retriever(vectordb=client)
    docs = [Document(page_content="Hello world", metadata={"source": "bar"}),
            Document(page_content="Hello", metadata={"source": "foo"})]
    ids = retriever.gen_doc_ids(docs)
    assert len(ids) == len(docs)
    assert all(isinstance(id, str) for id in ids)
    del client, retriever


def test_retriever_split_docs():
    pass


def test_retriever_add_docs():
    client = DeepLakeClient(collection_name="test_retriever")
    retriever = Retriever(vectordb=client)
    # small enough docs to not split.
    docs = [Document(page_content=
                     """And so on this rainbow day, with storms all around them, and blue sky
                 above, they rode only as far as the valley. But from there, before they
                 turned to go back, the monuments appeared close, and they loomed
                 grandly with the background of purple bank and creamy cloud and shafts
                 of golden lightning. They seemed like sentinels--guardians of a great
                 and beautiful love born under their lofty heights, in the lonely
                 silence of day, in the star-thrown shadow of night. They were like that
                 love. And they held Lucy and Slone, calling every day, giving a
                 nameless and tranquil content, binding them true to love, true to the
                 sage and the open, true to that wild upland home""", metadata={"source": "test_doc_1"}),
            Document(page_content=
                     """Slone and Lucy never rode down so far as the stately monuments, though
                     these held memories as hauntingly sweet as others were poignantly
                     bitter. Lucy never rode the King again. But Slone rode him, learned to
                     love him. And Lucy did not race any more. When Slone tried to stir in
                     her the old spirit all the response he got was a wistful shake of head
                     or a laugh that hid the truth or an excuse that the strain on her
                     ankles from Joel Creech's lasso had never mended. The girl was
                     unutterably happy, but it was possible that she would never race a
                     horse again.""", metadata={"source": "test_doc_2"}),
            Document(page_content=
                     """Bostil wanted to be alone, to welcome the King, to lead him back to the
                 home corral, perhaps to hide from all eyes the change and the uplift
                 that would forever keep him from wronging another man.
             
                 The late rains came and like magic, in a few days, the sage grew green
                 and lustrous and fresh, the gray turning to purple.
             
                 Every morning the sun rose white and hot in a blue and cloudless sky.
                 And then soon the horizon line showed creamy clouds that rose and
                 spread and darkened. Every afternoon storms hung along the ramparts and
                 rainbows curved down beautiful and ethereal. The dim blackness of the
                 storm-clouds was split to the blinding zigzag of lightning, and the
                 thunder rolled and boomed, like the Colorado in flood.""", metadata={"source": "test_doc_3"})
            ]
    ids = retriever.gen_doc_ids(docs)
    retriever.add_docs(docs)
    retrieved = retriever.docstore.mget(ids)
    assert len(retrieved) == len(ids)
    for ret, doc in zip(retrieved, docs):
        assert ret.metadata == doc.metadata
    del client, retriever


def test_retriever_aadd_docs():
    pass

# # add code folder to sys path
# import os
# from pathlib import Path
#
# from grag.components.multivec_retriever import Retriever
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
#
# # %%%
# # data_path = "data/pdf/9809"
# data_path = Path(os.getcwd()).parents[1] / 'data' / 'Gutenberg' / 'txt'  # "data/Gutenberg/txt"
# # %%
# retriever = Retriever(top_k=3)
#
# new_docs = True
#
# if new_docs:
#     # loading text files from data_path
#     loader = DirectoryLoader(data_path,
#                              glob="*.txt",
#                              loader_cls=TextLoader,
#                              show_progress=True,
#                              use_multithreading=True)
#     print('Loading Files: ')
#     docs = loader.load()
#     # %%
#     # limit docs for testing
#     docs = docs[:100]
#
#     # %%
#     # adding chunks and parent doc
#     retriever.add_docs(docs)
# # %%
# # testing retrival
# query = 'Thomas H. Huxley'
#
# # Retrieving the 3 most relevant small chunk
# chunk_result = retriever.get_chunk(query)
#
# # Retrieving the most relevant document
# doc_result = retriever.get_doc(query)
#
# # Ensuring that the length of chunk is smaller than length of doc
# chunk_len = [len(chunk.page_content) for chunk in chunk_result]
# print(f'Length of chunks : {chunk_len}')
# doc_len = [len(doc.page_content) for doc in doc_result]
# print(f'Length of doc    : {doc_len}')
# len_test = [c_len < d_len for c_len, d_len in zip(chunk_len, doc_len)]
# print(f'Is len of chunk less than len of doc?: {len_test} ')
#
# # Ensuring both the chunk and document match the source
# chunk_sources = [chunk.metadata['source'] for chunk in chunk_result]
# doc_sources = [doc.metadata['source'] for doc in doc_result]
# source_test = [source[0] == source[1] for source in zip(chunk_sources, doc_sources)]
# print(f'Does source of chunk and doc match? : {source_test}')
#
# # # Ensuring both the chunk and document match the source
# # source_test = chunk_result.metadata['source'] == doc_result.metadata['source']
# # print(f'Does source of chunk and doc match? : {source_test}')
# # # Ensuring that the length of chunk is smaller than length of doc
# # len_test = chunk_len <= doc_len
# # print(f'Is len of chunk less than len of doc?: {len_test} ')
# # %%
