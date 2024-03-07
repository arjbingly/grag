## Repo Structure

___

```
.
├── Readme.md
├── data
│   └── pdf
├── documentation
│   ├── AWS_Setup_Nvidia_Driver_Install.md
│   ├── AWS_Setup_Python_Env.md
│   ├── Building an effective RAG app.md
│   ├── Data Sources.md
│   ├── basic_RAG_pipeline.drawio.svg
│   └── challenges.md
├── llm_quantize
│   ├── quantize.py
│   └── readme.md
├── models
├── projects
│   ├── Basic-RAG
│   │   ├── BasicRAG-ingest_data.py
│   │   ├── BasicRAG_v1.py
│   │   ├── BasicRAG_v2.py
│   │   ├── RAG-Piplines.md
│   │   ├── prompts
│   │   │   ├── Llama-2_QA-refine_1.json
│   │   │   ├── Llama-2_QA_1.json
│   │   │   ├── Mixtral-8x7b_QA_1.json
│   │   │   └── prompt_structure.json
│   │   └── tests
│   │       ├── BasicRAG_v1_test.py
│   │       └── BasicRAG_v2_test.py
│   └── Retriver-GUI
│       └── retriever_app.py
├── proposal
│   └── proposal.md
├── requirements.yml
└── src
    ├── Readme.md
    ├── __init__.py
    ├── components
    │   ├── __init__.py
    │   ├── chroma_client.py
    │   ├── config.py
    │   ├── embedding.py
    │   ├── llm.py
    │   ├── multivec_retriever.py
    │   ├── parse_pdf.py
    │   ├── text_splitter.py
    │   └── utils.py
    ├── scripts
    │   ├── reset_chroma.sh
    │   ├── reset_store.sh
    │   └── run_chroma.sh
    ├── tests
    │   ├── chroma_add_test.py
    │   ├── chroma_async_test.py
    │   ├── embedding_test.py
    │   ├── llm_test.py
    │   ├── multivec_retriever_test.py
    │   └── parse_pdf_test.py
    └── utils
        └── txt_data_ingest.py
```

---

## Project Overview

A ready to deploy RAG pipeline for document retrival.

---

## To get started

To run the projects make sure below instructions are followed.

Moreover, further customization can be made on the config file, `src/config.ini`.

### Requirements

Use conda package manager to create an environment using the `requirements.yml`  
`conda env create -f requirements.yml`

_If you need help installing conda, refer to [AWS_Setup_Python_Env](./documentation/AWS_Setup_Python_Env.md)_

Required packages includes (but not limited to):

- PyTorch
- LangChain
- Chroma
- Unstructured.io
- sentence-embedding
- instructor-embedding

### LLM Models

- **To run models locally** refer the [LLM Quantize Readme](./llm_quantize/readme.md) for details on downloading and
  quantizing LLM models.
- **To run models from Huggingface**, change the `model_name` under `llm` in `src/config.ini` to the huggingface
  repo-id (If
  models are not public, make sure you have the auth token).

**Tested models:**

1. Llama-2 7B, 13B
2. Mixtral 8x7B

### Data

The project utilized ArXiv papers pdfs. Refer to [ArXiv Bulk Data](https://info.arxiv.org/help/bulk_data/index.html) for
details on how to download.

```
├── data
│   ├── pdf
```

**Make sure to specify `data_path` under `data` in `src/config.ini`**

### Vector Database (Chroma) - Data Ingestion

The vector database of choice os [Chroma](https://www.trychroma.com). Though most vector databases supported by
LangChain should work with minimal changes.

For ingesting data to the vector db:

- To run Chroma locally, move to `src/scripts` then run `source run_chroma.sh`. This by default runs on port 8000.
- If Chroma is not run locally, change `host` and `port` under `chroma` in `src/config.ini`.
- By default, the embedding model is `instructor-xl`. Can be changed by changing `embedding_type` and `embedding_model`
  in `src/config.ini'. Any huggingface embeddings can be used.
- To add files to Chroma, run `projects/Basic-RAG/BasicRAG-ingest_data.py`. Make sure that the data-path in the python
  file is correct.

---

## Other Features

### PDF Parser

- The pdf parser is implemented using [Unstructured.io](https://unstructured.io).
- It effectively parses any pdf including OCR documents and categorises all elements including tables and images.
- Contextual text parsing, it ensures that the chunking process does not separate items like list items, and keeps
  titles intact with text.
- Tables are not chunked.

### Multi Vector Retriever

- It enables to easily retrieve not only the most similar chunks (to a query) but easily retrieve the source document.

---

## Projects

### 1. Retriever GUI

A simple GUI for retrieving documents and viewing config of the vector database

To run: `streamlit run projects/retriver_app.py -server.port=8888`

### 2. BasicRAG

Refer to [BasicRAG/README.md](projects/BasicRAG/README.md)

___

## Instruction for facing GitHub page

- Every group need to summarize the whole project to a one-minute video.
- It should showcase the main points.

![Watch the video](../Sample_Capstone/demo/fig/demo.gif)
