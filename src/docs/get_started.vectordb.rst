Vector Stores
===============

# Explain what a vector store is...

Supported Vector Stores
########################

Currently supported vectorstores are:

1. Chroma
2. Deeplake

Chroma
*******
Since Chroma is a server-client based vector database, make sure to run the server.

* To run Chroma locally, either use move to `src/scripts` then run `source run_chroma.sh` or refer to
  `Running Chroma in ClientServer <https://docs.trychroma.com/usage-guide#running-chroma-in-clientserver-mode>`_.
  This by default runs on port 8000.
* If Chroma is not run locally, change `host` and `port` under `chroma` in `src/config.ini`, or provide the arguments
  explicitly.


Embeddings
###########

* By default, the embedding model is `instructor-xl`. Can be changed by changing `embedding_type` and `embedding_model`
  in `src/config.ini` or providing the arguments explicitly.
* Any huggingface embeddings can be used.

Data Ingestion
###############
::

    client = DeepLakeClient() # Any vectordb client
    retriever = Retriever(vectordb=client)


    dir_path = Path(__file__).parents[2] # path to folder containing pdf files


    retriever.ingest(dir_path)
