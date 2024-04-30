Vector Stores
===============

Vector store or vector database is a type of database that stores data in high-dimensional vectors. 
This is a crucial component of RAG, storing embeddings for both retrieval and generation processes.

Supported Vector Stores
########################

Currently supported vectorstores are:

1. Chroma
2. DeepLake

Chroma
*******
Since Chroma is a server-client based vector database, make sure to run the server.

* To run Chroma locally, either:
  
  - Move to `src/scripts` then run ``source run_chroma.sh`` OR
  
  - Refer to `Running Chroma in ClientServer <https://docs.trychroma.com/usage-guide#running-chroma-in-clientserver-mode>`_.
  This by default runs on port 8000.
  
* If Chroma is not run locally, change ``host`` and ``port`` under ``chroma`` in `src/config.ini`, or provide the arguments
  explicitly.

Once you have chroma running, just use the Chroma Client class.

DeepLake
*********
Since DeepLake is not a server based vector store, it is much easier to get started.

Just make sure you have DeepLake installed and use the DeepLake Client class.


Embeddings
###########

* By default, the embedding model is `instructor-xl`. Can be changed by changing ``embedding_type`` and ``embedding_model``
  in `src/config.ini` or providing the arguments explicitly.
* Any huggingface embeddings can be used.

Data Ingestion
###############

For more details on data ingestion, refer to our `cookbook <https://github.com/arjbingly/Capstone_5/blob/main/cookbook/Basic-RAG/README.md>`_.

::

    client = DeepLakeClient() # Any vectordb client
    retriever = Retriever(vectordb=client)


    dir_path = Path(__file__).parents[2] # path to folder containing pdf files


    retriever.ingest(dir_path)
