# GRAG (note: specify the abbreviation)

![GitHub License](https://img.shields.io/github/license/arjbingly/Capstone_5)
![Linting](https://img.shields.io/github/actions/workflow/status/arjbingly/Capstone_5/linting.yml?label=Docs&labelColor=yellow)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/arjbingly/Capstone_5/build_linting.yml?label=Linting)
![Static Badge](https://img.shields.io/badge/Tests-passing-darggreen)
![Static Badge](https://img.shields.io/badge/docstring%20style-google-yellow)
![Static Badge](https://img.shields.io/badge/linter%20-ruff-yellow)
![Static Badge](https://img.shields.io/badge/buildstyle-hatchling-purple?labelColor=white)
![Static Badge](https://img.shields.io/badge/codestyle-pyflake-purple?labelColor=white)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/arjbingly/Capstone_5)

(note: add overview on what the purpose of this project is here. Talk briefly about RAG. Maybe copy from the proposal)

<figure>
    <img src="documentation/basic_RAG_pipeline.png" alt="Diagram of a basic RAG pipeline">
    <figcaption style="text-align: center;"
    >Diagram of a basic RAG pipeline</figcaption>
</figure>

## Table of Content

- [Project Overview](#project-overview--change-this-to-features--)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [LLM Models](#llm-models)
  - [Data](#data)
  - [Supported Vector Databases](#supported-vector-databases)
    - [Embeddings](#embeddings)
  - [Data Ingestion](#data-ingestion)
- [Main Features](#main-features)
  - [1. PDF Parser](#1-pdf-parser)
  - [2. Multi-Vector Retriever](#2-multi-vector-retriever)
  - [3. BasicRAG](#3-basicrag)
- [GUI](#gui)
  - [1. Retriever GUI](#1-retriever-gui)
  - [2. BasicRAG GUI](#2-basicrag-gui)
- [Demo](#demo)
- [Repo Structure](#repo-structure)

## Project Overview

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

### Requirements

Required packages to install includes (_refer to [pyproject.toml](pyproject.toml)_):

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
│   ├── pdf
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

```
.
├── LICENSE
├── README.md
├── ci
│   ├── Jenkinsfile
│   ├── env_test.py
│   ├── modify_config.py
│   └── unlock_deeplake.py
├── cookbook
│   ├── Basic-RAG
│   │   ├── BasicRAG_CustomPrompt.py
│   │   ├── BasicRAG_FewShotPrompt.py
│   │   ├── BasicRAG_ingest.py
│   │   ├── BasicRAG_refine.py
│   │   ├── BasicRAG_stuff.py
│   │   ├── RAG-PIPELINES.md
│   │   └── README.md
│   └── Retriver-GUI
│       └── retriever_app.py
├── demo
│   ├── Readme.md
│   └── fig
│       ├── demo.gif
│       └── video.mp4
├── documentation
│   ├── AWS_Setup_Nvidia_Driver_Install.md
│   ├── AWS_Setup_Python_Env.md
│   ├── Building an effective RAG app.md
│   ├── Data Sources.md
│   ├── basic_RAG_pipeline.drawio.svg
│   └── challenges.md
├── full_report
│   ├── Latex_report
│   │   ├── File_Setup.tex
│   │   ├── Sample_Report.pdf
│   │   ├── Sample_Report.tex
│   │   ├── fig
│   │   │   ├── GW_logo-eps-converted-to.pdf
│   │   │   ├── GW_logo.eps
│   │   │   ├── ascent-archi.pdf
│   │   │   ├── certificates-log-archi.pdf
│   │   │   ├── nyush-logo.jpeg
│   │   │   └── perf-plot-1.pdf
│   │   └── references.bib
│   ├── Markdown_Report
│   ├── Readme.md
│   └── Word_Report
│       ├── Sample_Report.docx
│       └── Sample_Report.pdf
├── llm_quantize
│   └── README.md
├── presentation
│   └── Readme.md
├── proposal
│   └── proposal.md
├── pyproject.toml
├── requirements.yml
├── research_paper
│   ├── Latex
│   │   ├── Fig
│   │   │   ├── narxnet1-eps-converted-to.pdf
│   │   │   └── narxnet1.eps
│   │   ├── Paper_Temp.pdf
│   │   ├── Paper_Temp.tex
│   │   └── mybib.bib
│   ├── Readme.md
│   └── Word
│       └── Conference-template-A4.doc
└── src
    ├── __init__.py
    ├── config.ini
    ├── grag
    │   ├── __about__.py
    │   ├── __init__.py
    │   ├── components
    │   │   ├── __init__.py
    │   │   ├── embedding.py
    │   │   ├── llm.py
    │   │   ├── multivec_retriever.py
    │   │   ├── parse_pdf.py
    │   │   ├── prompt.py
    │   │   ├── text_splitter.py
    │   │   ├── utils.py
    │   │   └── vectordb
    │   │       ├── __init__.py
    │   │       ├── base.py
    │   │       ├── chroma_client.py
    │   │       └── deeplake_client.py
    │   ├── prompts
    │   │   ├── Llama-2_QA-refine_1.json
    │   │   ├── Llama-2_QA_1.json
    │   │   ├── Mixtral_QA_1.json
    │   │   ├── __init__.py
    │   │   └── matcher.json
    │   ├── quantize
    │   │   ├── __init__.py
    │   │   ├── quantize.py
    │   │   └── utils.py
    │   └── rag
    │       ├── __init__.py
    │       └── basic_rag.py
    ├── scripts
    │   ├── reset_chroma.sh
    │   ├── reset_store.sh
    │   └── run_chroma.sh
    └── tests
        ├── README.md
        ├── __init__.py
        ├── components
        │   ├── __init__.py
        │   ├── embedding_test.py
        │   ├── llm_test.py
        │   ├── multivec_retriever_test.py
        │   ├── parse_pdf_test.py
        │   ├── prompt_test.py
        │   ├── utils_test.py
        │   └── vectordb
        │       ├── __init__.py
        │       ├── chroma_client_test.py
        │       └── deeplake_client_test.py
        ├── quantize
        │   ├── __init__.py
        │   └── quantize_test.py
        └── rag
            ├── __init__.py
            └── basic_rag_test.py
```

---
