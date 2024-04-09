## Repo Structure

___

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

## Project Overview

A ready to deploy RAG pipeline for document retrival.

---

## To get started

To run the projects make sure below instructions are followed.

Moreover, further customization can be made on the config file, `src/config.ini`.

### Requirements

Use conda package manager to create an environment using the `requirements.yml`  
`conda env create -f requirements.yml`

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

### 2. BasicRAG

Refer to [BasicRAG/README.md](./projects/Basic-RAG/README.md)

___

## Instruction for facing GitHub page

- Every group need to summarize the whole project to a one-minute video.
- It should showcase the main points.

![Watch the video](../Sample_Capstone/demo/fig/demo.gif)
