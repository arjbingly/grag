# GRAG (note: specify the abbreviation)

(note: insert the interactive tags here, ask Amir about copyright and add the tags)

(note: add overview on what the purpose of this project is here. Talk briefly about RAG. Maybe copy from the proposal)

Need to include steps or a diagram of steps here.

## Table of Content

## Project Overview (change this to Features?)

- A ready to deploy RAG pipeline for document retrival.
- Basic GUI _(Under Development)_
- Evaluation Suite _(Under Development)_
- RAG enhancement using Graphs _(Under Development)_

---

## Getting Started

To run the projects, make sure the instructions below are followed.

Further customization can be made on the config file, `src/config.ini`.

- `git clone` the repository
- `pip install .` from the repository (note: add - then change directory to the cloned repo)
- _For Dev:_ `pip install -e .`

If you need to set up `conda` in your environment,

### Requirements

Required packages to install includes (_refer to [pyproject.toml](pyproject.toml)_):

- PyTorch
- LangChain
- Chroma
- Unstructured.io
- sentence-embedding
- instructor-embedding

### LLM Models

To quantize model, run:
`python -m grag.quantize.quantize`

For more details, go to [.\llm_quantize\readme.md](.\llm_quantize\readme.md)
**Tested models:**

1. Llama-2 7B, 13B
2. Mixtral 8x7B
3. Gemma 7B

**Model Compatibility**

Refer to [llama.cpp](https://github.com/ggerganov/llama.cpp) Supported Models (under Description) for list of compatible models.

### Data

Any PDF can be used for this project. We personally tested the project using ArXiv papers. Refer [ArXiv Bulk Data](https://info.arxiv.org/help/bulk_data/index.html) for
details on how to download.

```
├── data
│   ├── pdf
```

**Make sure to specify `data_path` under `data` in `src/config.ini`**

### Supported Vector Databases

**1. [Chroma](https://www.trychroma.com)**

Since Chroma is a server-client based vector database, make sure to run the server.

- To run Chroma locally, move to `src/scripts` then run `source run_chroma.sh`. This by default runs on port 8000.
- If Chroma is not run locally, change `host` and `port` under `chroma` in `src/config.ini`.

**2. [Deeplake](https://www.deeplake.ai/)**

#### Embeddings

- By default, the embedding model is `instructor-xl`. Can be changed by changing `embedding_type` and `embedding_model`
  in `src/config.ini'. Any huggingface embeddings can be used.

### Data Ingestion

For ingesting data to the vector db:

```
client = DeepLakeClient() # Any vectordb client
retriever = Retriever(vectordb=client)

dir_path = Path(__file__).parents[2] # path to folder containing pdf files

retriever.ingest(dir_path)
```

Refer to ['cookbook/basicRAG/BasicRAG_ingest'](./cookbook/basicRAG/BasicRAG_ingest)

---

## Main Features

### 1. PDF Parser

(note: need to rewrite this. Under contruction: test suites and documentation for every iteration)

- The pdf parser is implemented using [Unstructured.io](https://unstructured.io).
- It effectively parses any pdf including OCR documents and categorises all elements including tables and images.
- Enables contextual text parsing: it ensures that the chunking process does not separate items like list items, and keeps titles together with text.
- Tables are not chunked.

### 2. Multi-Vector Retriever

- It easily retrieves not only the most similar chunks (to a query) but also the source document of the chunks.

### 3. BasicRAG

Refer to [BasicRAG/README.md](./cookbook/Basic-RAG/README.md)
(note: fix the RAGPipeline.md link)

---

## GUI

### 1. Retriever GUI

A simple GUI for retrieving documents and viewing config of the vector database.

To run: `streamlit run projects/retriver_app.py -server.port=8888`

### 2. BasicRAG GUI

Under development.

---

## Demo

(to be added)
![Watch the video](../Sample_Capstone/demo/fig/demo.gif)

## Repo Structure

---

(note: update the repo structure)

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
