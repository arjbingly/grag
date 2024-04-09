<h1 align="center">Graph Retrieval-Augmented Generation - GRAG</h1>

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![Static Badge](https://img.shields.io/badge/docstring%20style-google-yellow)
![Static Badge](https://img.shields.io/badge/linter%20-ruff-yellow)
![Linting](https://img.shields.io/github/actions/workflow/status/arjbingly/Capstone_5/ruff_linting.yml?label=Docs&labelColor=yellow)
![Static Badge](https://img.shields.io/badge/buildstyle-hatchling-purple?labelColor=white)
![Static Badge](https://img.shields.io/badge/codestyle-pyflake-purple?labelColor=white)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/arjbingly/Capstone_5)

This GitRepo provides an open-sourced implementation of a Retrival-Augmented Generation pipeline, using a graph data structure in place of a vector database.

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

```
.
├── LICENSE
├── README.md
├── ci
│   ├── Jenkinsfile
│   ├── env_test.py
│   ├── modify_config.py
│   └── unlock_deeplake.py
├── code
│   ├── components
│   │   └── __pycache__
│   │       └── parse_pdf.cpython-38.pyc
│   └── test case data
│       ├── docs
│       ├── html
│       ├── md
│       └── txt
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
├── data
│   └── test
│       ├── pdf
│       │   ├── SSRN-id801185.pdf
│       │   └── pdf
│       │       └── 0001
│       │           ├── 0001001v1.pdf
│       │           ├── 0001002v1.pdf
│       │           ├── 0001004v1.pdf
│       │           ├── 0001005v1.pdf
│       │           └── 0001006v1.pdf
│       └── pdf.tar
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
│   │   ├── Capstone5_report_v2.log
│   │   ├── Capstone5_report_v2.tex
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
├── models
│   └── ggml-model-Q5_K_M.gguf
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
├── src
│   ├── __init__.py
│   ├── config.ini
│   ├── docs
│   │   ├── Makefile
│   │   ├── _build
│   │   │   ├── doctrees
│   │   │   │   ├── environment.pickle
│   │   │   │   ├── grag.components.doctree
│   │   │   │   ├── grag.components.vectordb.doctree
│   │   │   │   ├── grag.doctree
│   │   │   │   ├── grag.prompts.doctree
│   │   │   │   ├── grag.quantize.doctree
│   │   │   │   ├── grag.rag.doctree
│   │   │   │   ├── index.doctree
│   │   │   │   └── modules.doctree
│   │   │   └── html
│   │   │       ├── _modules
│   │   │       │   ├── grag
│   │   │       │   │   ├── components
│   │   │       │   │   │   ├── embedding.html
│   │   │       │   │   │   ├── llm.html
│   │   │       │   │   │   ├── multivec_retriever.html
│   │   │       │   │   │   ├── parse_pdf.html
│   │   │       │   │   │   ├── prompt.html
│   │   │       │   │   │   ├── text_splitter.html
│   │   │       │   │   │   ├── utils.html
│   │   │       │   │   │   └── vectordb
│   │   │       │   │   │       ├── base.html
│   │   │       │   │   │       ├── chroma_client.html
│   │   │       │   │   │       └── deeplake_client.html
│   │   │       │   │   ├── quantize
│   │   │       │   │   │   └── utils.html
│   │   │       │   │   └── rag
│   │   │       │   │       └── basic_rag.html
│   │   │       │   └── index.html
│   │   │       ├── _sources
│   │   │       │   ├── grag.components.rst.txt
│   │   │       │   ├── grag.components.vectordb.rst.txt
│   │   │       │   ├── grag.prompts.rst.txt
│   │   │       │   ├── grag.quantize.rst.txt
│   │   │       │   ├── grag.rag.rst.txt
│   │   │       │   ├── grag.rst.txt
│   │   │       │   ├── index.rst.txt
│   │   │       │   └── modules.rst.txt
│   │   │       ├── _static
│   │   │       │   ├── _sphinx_javascript_frameworks_compat.js
│   │   │       │   ├── autoclasstoc.css
│   │   │       │   ├── basic.css
│   │   │       │   ├── css
│   │   │       │   │   ├── badge_only.css
│   │   │       │   │   ├── fonts
│   │   │       │   │   │   ├── Roboto-Slab-Bold.woff
│   │   │       │   │   │   ├── Roboto-Slab-Bold.woff2
│   │   │       │   │   │   ├── Roboto-Slab-Regular.woff
│   │   │       │   │   │   ├── Roboto-Slab-Regular.woff2
│   │   │       │   │   │   ├── fontawesome-webfont.eot
│   │   │       │   │   │   ├── fontawesome-webfont.svg
│   │   │       │   │   │   ├── fontawesome-webfont.ttf
│   │   │       │   │   │   ├── fontawesome-webfont.woff
│   │   │       │   │   │   ├── fontawesome-webfont.woff2
│   │   │       │   │   │   ├── lato-bold-italic.woff
│   │   │       │   │   │   ├── lato-bold-italic.woff2
│   │   │       │   │   │   ├── lato-bold.woff
│   │   │       │   │   │   ├── lato-bold.woff2
│   │   │       │   │   │   ├── lato-normal-italic.woff
│   │   │       │   │   │   ├── lato-normal-italic.woff2
│   │   │       │   │   │   ├── lato-normal.woff
│   │   │       │   │   │   └── lato-normal.woff2
│   │   │       │   │   └── theme.css
│   │   │       │   ├── doctools.js
│   │   │       │   ├── documentation_options.js
│   │   │       │   ├── file.png
│   │   │       │   ├── jquery.js
│   │   │       │   ├── js
│   │   │       │   │   ├── badge_only.js
│   │   │       │   │   ├── html5shiv-printshiv.min.js
│   │   │       │   │   ├── html5shiv.min.js
│   │   │       │   │   └── theme.js
│   │   │       │   ├── language_data.js
│   │   │       │   ├── minus.png
│   │   │       │   ├── plus.png
│   │   │       │   ├── pygments.css
│   │   │       │   ├── searchtools.js
│   │   │       │   └── sphinx_highlight.js
│   │   │       ├── genindex.html
│   │   │       ├── grag.components.html
│   │   │       ├── grag.components.vectordb.html
│   │   │       ├── grag.html
│   │   │       ├── grag.prompts.html
│   │   │       ├── grag.quantize.html
│   │   │       ├── grag.rag.html
│   │   │       ├── index.html
│   │   │       ├── modules.html
│   │   │       ├── objects.inv
│   │   │       ├── py-modindex.html
│   │   │       ├── search.html
│   │   │       └── searchindex.js
│   │   ├── conf.py
│   │   ├── grag.components.rst
│   │   ├── grag.components.vectordb.rst
│   │   ├── grag.prompts.rst
│   │   ├── grag.quantize.rst
│   │   ├── grag.rag.rst
│   │   ├── grag.rst
│   │   ├── index.rst
│   │   ├── make.bat
│   │   └── modules.rst
│   ├── grag
│   │   ├── __about__.py
│   │   ├── __init__.py
│   │   ├── components
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   ├── llm.py
│   │   │   ├── multivec_retriever.py
│   │   │   ├── parse_pdf.py
│   │   │   ├── prompt.py
│   │   │   ├── text_splitter.py
│   │   │   ├── utils.py
│   │   │   └── vectordb
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       ├── chroma_client.py
│   │   │       └── deeplake_client.py
│   │   ├── prompts
│   │   │   ├── Llama-2_QA-refine_1.json
│   │   │   ├── Llama-2_QA_1.json
│   │   │   ├── Mixtral_QA_1.json
│   │   │   ├── __init__.py
│   │   │   └── matcher.json
│   │   ├── quantize
│   │   │   ├── __init__.py
│   │   │   ├── quantize.py
│   │   │   └── utils.py
│   │   └── rag
│   │       ├── __init__.py
│   │       └── basic_rag.py
│   ├── scripts
│   │   ├── reset_chroma.sh
│   │   ├── reset_store.sh
│   │   └── run_chroma.sh
│   └── tests
│       ├── README.md
│       ├── __init__.py
│       ├── components
│       │   ├── __init__.py
│       │   ├── embedding_test.py
│       │   ├── llm_test.py
│       │   ├── multivec_retriever_test.py
│       │   ├── parse_pdf_test.py
│       │   ├── prompt_test.py
│       │   ├── utils_test.py
│       │   └── vectordb
│       │       ├── __init__.py
│       │       ├── chroma_client_test.py
│       │       └── deeplake_client_test.py
│       ├── quantize
│       │   ├── __init__.py
│       │   └── quantize_test.py
│       └── rag
│           ├── __init__.py
│           └── basic_rag_test.py
└── venv
    ├── Lib
    │   └── site-packages
    │       ├── __pycache__
    │       │   └── _virtualenv.cpython-39.pyc
    │       ├── _distutils_hack
    │       │   ├── __init__.py
    │       │   ├── __pycache__
    │       │   │   └── __init__.cpython-39.pyc
    │       │   └── override.py
    │       ├── _virtualenv.pth
    │       ├── _virtualenv.py
    │       ├── distutils-precedence.pth
    │       ├── pip
    │       │   ├── __init__.py
    │       │   ├── __main__.py
    │       │   ├── __pip-runner__.py
    │       │   ├── _internal
    │       │   │   ├── __init__.py
    │       │   │   ├── build_env.py
    │       │   │   ├── cache.py
    │       │   │   ├── cli
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── autocompletion.py
    │       │   │   │   ├── base_command.py
    │       │   │   │   ├── cmdoptions.py
    │       │   │   │   ├── command_context.py
    │       │   │   │   ├── main.py
    │       │   │   │   ├── main_parser.py
    │       │   │   │   ├── parser.py
    │       │   │   │   ├── progress_bars.py
    │       │   │   │   ├── req_command.py
    │       │   │   │   ├── spinners.py
    │       │   │   │   └── status_codes.py
    │       │   │   ├── commands
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── cache.py
    │       │   │   │   ├── check.py
    │       │   │   │   ├── completion.py
    │       │   │   │   ├── configuration.py
    │       │   │   │   ├── debug.py
    │       │   │   │   ├── download.py
    │       │   │   │   ├── freeze.py
    │       │   │   │   ├── hash.py
    │       │   │   │   ├── help.py
    │       │   │   │   ├── index.py
    │       │   │   │   ├── inspect.py
    │       │   │   │   ├── install.py
    │       │   │   │   ├── list.py
    │       │   │   │   ├── search.py
    │       │   │   │   ├── show.py
    │       │   │   │   ├── uninstall.py
    │       │   │   │   └── wheel.py
    │       │   │   ├── configuration.py
    │       │   │   ├── distributions
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── base.py
    │       │   │   │   ├── installed.py
    │       │   │   │   ├── sdist.py
    │       │   │   │   └── wheel.py
    │       │   │   ├── exceptions.py
    │       │   │   ├── index
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── collector.py
    │       │   │   │   ├── package_finder.py
    │       │   │   │   └── sources.py
    │       │   │   ├── locations
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _distutils.py
    │       │   │   │   ├── _sysconfig.py
    │       │   │   │   └── base.py
    │       │   │   ├── main.py
    │       │   │   ├── metadata
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _json.py
    │       │   │   │   ├── base.py
    │       │   │   │   ├── importlib
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── _compat.py
    │       │   │   │   │   ├── _dists.py
    │       │   │   │   │   └── _envs.py
    │       │   │   │   └── pkg_resources.py
    │       │   │   ├── models
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── candidate.py
    │       │   │   │   ├── direct_url.py
    │       │   │   │   ├── format_control.py
    │       │   │   │   ├── index.py
    │       │   │   │   ├── installation_report.py
    │       │   │   │   ├── link.py
    │       │   │   │   ├── scheme.py
    │       │   │   │   ├── search_scope.py
    │       │   │   │   ├── selection_prefs.py
    │       │   │   │   ├── target_python.py
    │       │   │   │   └── wheel.py
    │       │   │   ├── network
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── auth.py
    │       │   │   │   ├── cache.py
    │       │   │   │   ├── download.py
    │       │   │   │   ├── lazy_wheel.py
    │       │   │   │   ├── session.py
    │       │   │   │   ├── utils.py
    │       │   │   │   └── xmlrpc.py
    │       │   │   ├── operations
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── build
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── build_tracker.py
    │       │   │   │   │   ├── metadata.py
    │       │   │   │   │   ├── metadata_editable.py
    │       │   │   │   │   ├── metadata_legacy.py
    │       │   │   │   │   ├── wheel.py
    │       │   │   │   │   ├── wheel_editable.py
    │       │   │   │   │   └── wheel_legacy.py
    │       │   │   │   ├── check.py
    │       │   │   │   ├── freeze.py
    │       │   │   │   ├── install
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── editable_legacy.py
    │       │   │   │   │   ├── legacy.py
    │       │   │   │   │   └── wheel.py
    │       │   │   │   └── prepare.py
    │       │   │   ├── pyproject.py
    │       │   │   ├── req
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── constructors.py
    │       │   │   │   ├── req_file.py
    │       │   │   │   ├── req_install.py
    │       │   │   │   ├── req_set.py
    │       │   │   │   └── req_uninstall.py
    │       │   │   ├── resolution
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── base.py
    │       │   │   │   ├── legacy
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   └── resolver.py
    │       │   │   │   └── resolvelib
    │       │   │   │       ├── __init__.py
    │       │   │   │       ├── base.py
    │       │   │   │       ├── candidates.py
    │       │   │   │       ├── factory.py
    │       │   │   │       ├── found_candidates.py
    │       │   │   │       ├── provider.py
    │       │   │   │       ├── reporter.py
    │       │   │   │       ├── requirements.py
    │       │   │   │       └── resolver.py
    │       │   │   ├── self_outdated_check.py
    │       │   │   ├── utils
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _log.py
    │       │   │   │   ├── appdirs.py
    │       │   │   │   ├── compat.py
    │       │   │   │   ├── compatibility_tags.py
    │       │   │   │   ├── datetime.py
    │       │   │   │   ├── deprecation.py
    │       │   │   │   ├── direct_url_helpers.py
    │       │   │   │   ├── distutils_args.py
    │       │   │   │   ├── egg_link.py
    │       │   │   │   ├── encoding.py
    │       │   │   │   ├── entrypoints.py
    │       │   │   │   ├── filesystem.py
    │       │   │   │   ├── filetypes.py
    │       │   │   │   ├── glibc.py
    │       │   │   │   ├── hashes.py
    │       │   │   │   ├── inject_securetransport.py
    │       │   │   │   ├── logging.py
    │       │   │   │   ├── misc.py
    │       │   │   │   ├── models.py
    │       │   │   │   ├── packaging.py
    │       │   │   │   ├── setuptools_build.py
    │       │   │   │   ├── subprocess.py
    │       │   │   │   ├── temp_dir.py
    │       │   │   │   ├── unpacking.py
    │       │   │   │   ├── urls.py
    │       │   │   │   ├── virtualenv.py
    │       │   │   │   └── wheel.py
    │       │   │   ├── vcs
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── bazaar.py
    │       │   │   │   ├── git.py
    │       │   │   │   ├── mercurial.py
    │       │   │   │   ├── subversion.py
    │       │   │   │   └── versioncontrol.py
    │       │   │   └── wheel_builder.py
    │       │   ├── _vendor
    │       │   │   ├── __init__.py
    │       │   │   ├── cachecontrol
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _cmd.py
    │       │   │   │   ├── adapter.py
    │       │   │   │   ├── cache.py
    │       │   │   │   ├── caches
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── file_cache.py
    │       │   │   │   │   └── redis_cache.py
    │       │   │   │   ├── compat.py
    │       │   │   │   ├── controller.py
    │       │   │   │   ├── filewrapper.py
    │       │   │   │   ├── heuristics.py
    │       │   │   │   ├── serialize.py
    │       │   │   │   └── wrapper.py
    │       │   │   ├── certifi
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __main__.py
    │       │   │   │   ├── cacert.pem
    │       │   │   │   └── core.py
    │       │   │   ├── chardet
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── big5freq.py
    │       │   │   │   ├── big5prober.py
    │       │   │   │   ├── chardistribution.py
    │       │   │   │   ├── charsetgroupprober.py
    │       │   │   │   ├── charsetprober.py
    │       │   │   │   ├── cli
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   └── chardetect.py
    │       │   │   │   ├── codingstatemachine.py
    │       │   │   │   ├── cp949prober.py
    │       │   │   │   ├── enums.py
    │       │   │   │   ├── escprober.py
    │       │   │   │   ├── escsm.py
    │       │   │   │   ├── eucjpprober.py
    │       │   │   │   ├── euckrfreq.py
    │       │   │   │   ├── euckrprober.py
    │       │   │   │   ├── euctwfreq.py
    │       │   │   │   ├── euctwprober.py
    │       │   │   │   ├── gb2312freq.py
    │       │   │   │   ├── gb2312prober.py
    │       │   │   │   ├── hebrewprober.py
    │       │   │   │   ├── jisfreq.py
    │       │   │   │   ├── johabfreq.py
    │       │   │   │   ├── johabprober.py
    │       │   │   │   ├── jpcntx.py
    │       │   │   │   ├── langbulgarianmodel.py
    │       │   │   │   ├── langgreekmodel.py
    │       │   │   │   ├── langhebrewmodel.py
    │       │   │   │   ├── langhungarianmodel.py
    │       │   │   │   ├── langrussianmodel.py
    │       │   │   │   ├── langthaimodel.py
    │       │   │   │   ├── langturkishmodel.py
    │       │   │   │   ├── latin1prober.py
    │       │   │   │   ├── mbcharsetprober.py
    │       │   │   │   ├── mbcsgroupprober.py
    │       │   │   │   ├── mbcssm.py
    │       │   │   │   ├── metadata
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   └── languages.py
    │       │   │   │   ├── sbcharsetprober.py
    │       │   │   │   ├── sbcsgroupprober.py
    │       │   │   │   ├── sjisprober.py
    │       │   │   │   ├── universaldetector.py
    │       │   │   │   ├── utf1632prober.py
    │       │   │   │   ├── utf8prober.py
    │       │   │   │   └── version.py
    │       │   │   ├── colorama
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── ansi.py
    │       │   │   │   ├── ansitowin32.py
    │       │   │   │   ├── initialise.py
    │       │   │   │   ├── win32.py
    │       │   │   │   └── winterm.py
    │       │   │   ├── distlib
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── compat.py
    │       │   │   │   ├── database.py
    │       │   │   │   ├── index.py
    │       │   │   │   ├── locators.py
    │       │   │   │   ├── manifest.py
    │       │   │   │   ├── markers.py
    │       │   │   │   ├── metadata.py
    │       │   │   │   ├── resources.py
    │       │   │   │   ├── scripts.py
    │       │   │   │   ├── t32.exe
    │       │   │   │   ├── t64-arm.exe
    │       │   │   │   ├── t64.exe
    │       │   │   │   ├── util.py
    │       │   │   │   ├── version.py
    │       │   │   │   ├── w32.exe
    │       │   │   │   ├── w64-arm.exe
    │       │   │   │   ├── w64.exe
    │       │   │   │   └── wheel.py
    │       │   │   ├── distro
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __main__.py
    │       │   │   │   └── distro.py
    │       │   │   ├── idna
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── codec.py
    │       │   │   │   ├── compat.py
    │       │   │   │   ├── core.py
    │       │   │   │   ├── idnadata.py
    │       │   │   │   ├── intranges.py
    │       │   │   │   ├── package_data.py
    │       │   │   │   └── uts46data.py
    │       │   │   ├── msgpack
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── exceptions.py
    │       │   │   │   ├── ext.py
    │       │   │   │   └── fallback.py
    │       │   │   ├── packaging
    │       │   │   │   ├── __about__.py
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _manylinux.py
    │       │   │   │   ├── _musllinux.py
    │       │   │   │   ├── _structures.py
    │       │   │   │   ├── markers.py
    │       │   │   │   ├── requirements.py
    │       │   │   │   ├── specifiers.py
    │       │   │   │   ├── tags.py
    │       │   │   │   ├── utils.py
    │       │   │   │   └── version.py
    │       │   │   ├── pep517
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _compat.py
    │       │   │   │   ├── build.py
    │       │   │   │   ├── check.py
    │       │   │   │   ├── colorlog.py
    │       │   │   │   ├── dirtools.py
    │       │   │   │   ├── envbuild.py
    │       │   │   │   ├── in_process
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   └── _in_process.py
    │       │   │   │   ├── meta.py
    │       │   │   │   └── wrappers.py
    │       │   │   ├── pkg_resources
    │       │   │   │   ├── __init__.py
    │       │   │   │   └── py31compat.py
    │       │   │   ├── platformdirs
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __main__.py
    │       │   │   │   ├── android.py
    │       │   │   │   ├── api.py
    │       │   │   │   ├── macos.py
    │       │   │   │   ├── unix.py
    │       │   │   │   ├── version.py
    │       │   │   │   └── windows.py
    │       │   │   ├── pygments
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __main__.py
    │       │   │   │   ├── cmdline.py
    │       │   │   │   ├── console.py
    │       │   │   │   ├── filter.py
    │       │   │   │   ├── filters
    │       │   │   │   │   └── __init__.py
    │       │   │   │   ├── formatter.py
    │       │   │   │   ├── formatters
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── _mapping.py
    │       │   │   │   │   ├── bbcode.py
    │       │   │   │   │   ├── groff.py
    │       │   │   │   │   ├── html.py
    │       │   │   │   │   ├── img.py
    │       │   │   │   │   ├── irc.py
    │       │   │   │   │   ├── latex.py
    │       │   │   │   │   ├── other.py
    │       │   │   │   │   ├── pangomarkup.py
    │       │   │   │   │   ├── rtf.py
    │       │   │   │   │   ├── svg.py
    │       │   │   │   │   ├── terminal.py
    │       │   │   │   │   └── terminal256.py
    │       │   │   │   ├── lexer.py
    │       │   │   │   ├── lexers
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── _mapping.py
    │       │   │   │   │   └── python.py
    │       │   │   │   ├── modeline.py
    │       │   │   │   ├── plugin.py
    │       │   │   │   ├── regexopt.py
    │       │   │   │   ├── scanner.py
    │       │   │   │   ├── sphinxext.py
    │       │   │   │   ├── style.py
    │       │   │   │   ├── styles
    │       │   │   │   │   └── __init__.py
    │       │   │   │   ├── token.py
    │       │   │   │   ├── unistring.py
    │       │   │   │   └── util.py
    │       │   │   ├── pyparsing
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── actions.py
    │       │   │   │   ├── common.py
    │       │   │   │   ├── core.py
    │       │   │   │   ├── diagram
    │       │   │   │   │   └── __init__.py
    │       │   │   │   ├── exceptions.py
    │       │   │   │   ├── helpers.py
    │       │   │   │   ├── results.py
    │       │   │   │   ├── testing.py
    │       │   │   │   ├── unicode.py
    │       │   │   │   └── util.py
    │       │   │   ├── requests
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __version__.py
    │       │   │   │   ├── _internal_utils.py
    │       │   │   │   ├── adapters.py
    │       │   │   │   ├── api.py
    │       │   │   │   ├── auth.py
    │       │   │   │   ├── certs.py
    │       │   │   │   ├── compat.py
    │       │   │   │   ├── cookies.py
    │       │   │   │   ├── exceptions.py
    │       │   │   │   ├── help.py
    │       │   │   │   ├── hooks.py
    │       │   │   │   ├── models.py
    │       │   │   │   ├── packages.py
    │       │   │   │   ├── sessions.py
    │       │   │   │   ├── status_codes.py
    │       │   │   │   ├── structures.py
    │       │   │   │   └── utils.py
    │       │   │   ├── resolvelib
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── compat
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   └── collections_abc.py
    │       │   │   │   ├── providers.py
    │       │   │   │   ├── reporters.py
    │       │   │   │   ├── resolvers.py
    │       │   │   │   └── structs.py
    │       │   │   ├── rich
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __main__.py
    │       │   │   │   ├── _cell_widths.py
    │       │   │   │   ├── _emoji_codes.py
    │       │   │   │   ├── _emoji_replace.py
    │       │   │   │   ├── _export_format.py
    │       │   │   │   ├── _extension.py
    │       │   │   │   ├── _inspect.py
    │       │   │   │   ├── _log_render.py
    │       │   │   │   ├── _loop.py
    │       │   │   │   ├── _palettes.py
    │       │   │   │   ├── _pick.py
    │       │   │   │   ├── _ratio.py
    │       │   │   │   ├── _spinners.py
    │       │   │   │   ├── _stack.py
    │       │   │   │   ├── _timer.py
    │       │   │   │   ├── _win32_console.py
    │       │   │   │   ├── _windows.py
    │       │   │   │   ├── _windows_renderer.py
    │       │   │   │   ├── _wrap.py
    │       │   │   │   ├── abc.py
    │       │   │   │   ├── align.py
    │       │   │   │   ├── ansi.py
    │       │   │   │   ├── bar.py
    │       │   │   │   ├── box.py
    │       │   │   │   ├── cells.py
    │       │   │   │   ├── color.py
    │       │   │   │   ├── color_triplet.py
    │       │   │   │   ├── columns.py
    │       │   │   │   ├── console.py
    │       │   │   │   ├── constrain.py
    │       │   │   │   ├── containers.py
    │       │   │   │   ├── control.py
    │       │   │   │   ├── default_styles.py
    │       │   │   │   ├── diagnose.py
    │       │   │   │   ├── emoji.py
    │       │   │   │   ├── errors.py
    │       │   │   │   ├── file_proxy.py
    │       │   │   │   ├── filesize.py
    │       │   │   │   ├── highlighter.py
    │       │   │   │   ├── json.py
    │       │   │   │   ├── jupyter.py
    │       │   │   │   ├── layout.py
    │       │   │   │   ├── live.py
    │       │   │   │   ├── live_render.py
    │       │   │   │   ├── logging.py
    │       │   │   │   ├── markup.py
    │       │   │   │   ├── measure.py
    │       │   │   │   ├── padding.py
    │       │   │   │   ├── pager.py
    │       │   │   │   ├── palette.py
    │       │   │   │   ├── panel.py
    │       │   │   │   ├── pretty.py
    │       │   │   │   ├── progress.py
    │       │   │   │   ├── progress_bar.py
    │       │   │   │   ├── prompt.py
    │       │   │   │   ├── protocol.py
    │       │   │   │   ├── region.py
    │       │   │   │   ├── repr.py
    │       │   │   │   ├── rule.py
    │       │   │   │   ├── scope.py
    │       │   │   │   ├── screen.py
    │       │   │   │   ├── segment.py
    │       │   │   │   ├── spinner.py
    │       │   │   │   ├── status.py
    │       │   │   │   ├── style.py
    │       │   │   │   ├── styled.py
    │       │   │   │   ├── syntax.py
    │       │   │   │   ├── table.py
    │       │   │   │   ├── terminal_theme.py
    │       │   │   │   ├── text.py
    │       │   │   │   ├── theme.py
    │       │   │   │   ├── themes.py
    │       │   │   │   ├── traceback.py
    │       │   │   │   └── tree.py
    │       │   │   ├── six.py
    │       │   │   ├── tenacity
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _asyncio.py
    │       │   │   │   ├── _utils.py
    │       │   │   │   ├── after.py
    │       │   │   │   ├── before.py
    │       │   │   │   ├── before_sleep.py
    │       │   │   │   ├── nap.py
    │       │   │   │   ├── retry.py
    │       │   │   │   ├── stop.py
    │       │   │   │   ├── tornadoweb.py
    │       │   │   │   └── wait.py
    │       │   │   ├── tomli
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _parser.py
    │       │   │   │   ├── _re.py
    │       │   │   │   └── _types.py
    │       │   │   ├── typing_extensions.py
    │       │   │   ├── urllib3
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _collections.py
    │       │   │   │   ├── _version.py
    │       │   │   │   ├── connection.py
    │       │   │   │   ├── connectionpool.py
    │       │   │   │   ├── contrib
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── _appengine_environ.py
    │       │   │   │   │   ├── _securetransport
    │       │   │   │   │   │   ├── __init__.py
    │       │   │   │   │   │   ├── bindings.py
    │       │   │   │   │   │   └── low_level.py
    │       │   │   │   │   ├── appengine.py
    │       │   │   │   │   ├── ntlmpool.py
    │       │   │   │   │   ├── pyopenssl.py
    │       │   │   │   │   ├── securetransport.py
    │       │   │   │   │   └── socks.py
    │       │   │   │   ├── exceptions.py
    │       │   │   │   ├── fields.py
    │       │   │   │   ├── filepost.py
    │       │   │   │   ├── packages
    │       │   │   │   │   ├── __init__.py
    │       │   │   │   │   ├── backports
    │       │   │   │   │   │   ├── __init__.py
    │       │   │   │   │   │   └── makefile.py
    │       │   │   │   │   └── six.py
    │       │   │   │   ├── poolmanager.py
    │       │   │   │   ├── request.py
    │       │   │   │   ├── response.py
    │       │   │   │   └── util
    │       │   │   │       ├── __init__.py
    │       │   │   │       ├── connection.py
    │       │   │   │       ├── proxy.py
    │       │   │   │       ├── queue.py
    │       │   │   │       ├── request.py
    │       │   │   │       ├── response.py
    │       │   │   │       ├── retry.py
    │       │   │   │       ├── ssl_.py
    │       │   │   │       ├── ssl_match_hostname.py
    │       │   │   │       ├── ssltransport.py
    │       │   │   │       ├── timeout.py
    │       │   │   │       ├── url.py
    │       │   │   │       └── wait.py
    │       │   │   ├── vendor.txt
    │       │   │   └── webencodings
    │       │   │       ├── __init__.py
    │       │   │       ├── labels.py
    │       │   │       ├── mklabels.py
    │       │   │       ├── tests.py
    │       │   │       └── x_user_defined.py
    │       │   └── py.typed
    │       ├── pip-22.3.1.dist-info
    │       │   ├── INSTALLER
    │       │   ├── LICENSE.txt
    │       │   ├── METADATA
    │       │   ├── RECORD
    │       │   ├── WHEEL
    │       │   ├── entry_points.txt
    │       │   └── top_level.txt
    │       ├── pip-22.3.1.virtualenv
    │       ├── pkg_resources
    │       │   ├── __init__.py
    │       │   ├── __pycache__
    │       │   │   └── __init__.cpython-39.pyc
    │       │   ├── _vendor
    │       │   │   ├── __init__.py
    │       │   │   ├── __pycache__
    │       │   │   │   ├── __init__.cpython-39.pyc
    │       │   │   │   └── appdirs.cpython-39.pyc
    │       │   │   ├── appdirs.py
    │       │   │   ├── importlib_resources
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _adapters.py
    │       │   │   │   ├── _common.py
    │       │   │   │   ├── _compat.py
    │       │   │   │   ├── _itertools.py
    │       │   │   │   ├── _legacy.py
    │       │   │   │   ├── abc.py
    │       │   │   │   ├── readers.py
    │       │   │   │   └── simple.py
    │       │   │   ├── jaraco
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __pycache__
    │       │   │   │   │   ├── __init__.cpython-39.pyc
    │       │   │   │   │   ├── context.cpython-39.pyc
    │       │   │   │   │   └── functools.cpython-39.pyc
    │       │   │   │   ├── context.py
    │       │   │   │   ├── functools.py
    │       │   │   │   └── text
    │       │   │   │       ├── __init__.py
    │       │   │   │       └── __pycache__
    │       │   │   │           └── __init__.cpython-39.pyc
    │       │   │   ├── more_itertools
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __pycache__
    │       │   │   │   │   ├── __init__.cpython-39.pyc
    │       │   │   │   │   ├── more.cpython-39.pyc
    │       │   │   │   │   └── recipes.cpython-39.pyc
    │       │   │   │   ├── more.py
    │       │   │   │   └── recipes.py
    │       │   │   ├── packaging
    │       │   │   │   ├── __about__.py
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __pycache__
    │       │   │   │   │   ├── __about__.cpython-39.pyc
    │       │   │   │   │   ├── __init__.cpython-39.pyc
    │       │   │   │   │   ├── _manylinux.cpython-39.pyc
    │       │   │   │   │   ├── _musllinux.cpython-39.pyc
    │       │   │   │   │   ├── _structures.cpython-39.pyc
    │       │   │   │   │   ├── markers.cpython-39.pyc
    │       │   │   │   │   ├── requirements.cpython-39.pyc
    │       │   │   │   │   ├── specifiers.cpython-39.pyc
    │       │   │   │   │   ├── tags.cpython-39.pyc
    │       │   │   │   │   ├── utils.cpython-39.pyc
    │       │   │   │   │   └── version.cpython-39.pyc
    │       │   │   │   ├── _manylinux.py
    │       │   │   │   ├── _musllinux.py
    │       │   │   │   ├── _structures.py
    │       │   │   │   ├── markers.py
    │       │   │   │   ├── requirements.py
    │       │   │   │   ├── specifiers.py
    │       │   │   │   ├── tags.py
    │       │   │   │   ├── utils.py
    │       │   │   │   └── version.py
    │       │   │   ├── pyparsing
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── __pycache__
    │       │   │   │   │   ├── __init__.cpython-39.pyc
    │       │   │   │   │   ├── actions.cpython-39.pyc
    │       │   │   │   │   ├── common.cpython-39.pyc
    │       │   │   │   │   ├── core.cpython-39.pyc
    │       │   │   │   │   ├── exceptions.cpython-39.pyc
    │       │   │   │   │   ├── helpers.cpython-39.pyc
    │       │   │   │   │   ├── results.cpython-39.pyc
    │       │   │   │   │   ├── testing.cpython-39.pyc
    │       │   │   │   │   ├── unicode.cpython-39.pyc
    │       │   │   │   │   └── util.cpython-39.pyc
    │       │   │   │   ├── actions.py
    │       │   │   │   ├── common.py
    │       │   │   │   ├── core.py
    │       │   │   │   ├── diagram
    │       │   │   │   │   └── __init__.py
    │       │   │   │   ├── exceptions.py
    │       │   │   │   ├── helpers.py
    │       │   │   │   ├── results.py
    │       │   │   │   ├── testing.py
    │       │   │   │   ├── unicode.py
    │       │   │   │   └── util.py
    │       │   │   └── zipp.py
    │       │   └── extern
    │       │       ├── __init__.py
    │       │       └── __pycache__
    │       │           └── __init__.cpython-39.pyc
    │       ├── setuptools
    │       │   ├── __init__.py
    │       │   ├── _deprecation_warning.py
    │       │   ├── _distutils
    │       │   │   ├── __init__.py
    │       │   │   ├── _collections.py
    │       │   │   ├── _functools.py
    │       │   │   ├── _macos_compat.py
    │       │   │   ├── _msvccompiler.py
    │       │   │   ├── archive_util.py
    │       │   │   ├── bcppcompiler.py
    │       │   │   ├── ccompiler.py
    │       │   │   ├── cmd.py
    │       │   │   ├── command
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _framework_compat.py
    │       │   │   │   ├── bdist.py
    │       │   │   │   ├── bdist_dumb.py
    │       │   │   │   ├── bdist_rpm.py
    │       │   │   │   ├── build.py
    │       │   │   │   ├── build_clib.py
    │       │   │   │   ├── build_ext.py
    │       │   │   │   ├── build_py.py
    │       │   │   │   ├── build_scripts.py
    │       │   │   │   ├── check.py
    │       │   │   │   ├── clean.py
    │       │   │   │   ├── config.py
    │       │   │   │   ├── install.py
    │       │   │   │   ├── install_data.py
    │       │   │   │   ├── install_egg_info.py
    │       │   │   │   ├── install_headers.py
    │       │   │   │   ├── install_lib.py
    │       │   │   │   ├── install_scripts.py
    │       │   │   │   ├── py37compat.py
    │       │   │   │   ├── register.py
    │       │   │   │   ├── sdist.py
    │       │   │   │   └── upload.py
    │       │   │   ├── config.py
    │       │   │   ├── core.py
    │       │   │   ├── cygwinccompiler.py
    │       │   │   ├── debug.py
    │       │   │   ├── dep_util.py
    │       │   │   ├── dir_util.py
    │       │   │   ├── dist.py
    │       │   │   ├── errors.py
    │       │   │   ├── extension.py
    │       │   │   ├── fancy_getopt.py
    │       │   │   ├── file_util.py
    │       │   │   ├── filelist.py
    │       │   │   ├── log.py
    │       │   │   ├── msvc9compiler.py
    │       │   │   ├── msvccompiler.py
    │       │   │   ├── py38compat.py
    │       │   │   ├── py39compat.py
    │       │   │   ├── spawn.py
    │       │   │   ├── sysconfig.py
    │       │   │   ├── text_file.py
    │       │   │   ├── unixccompiler.py
    │       │   │   ├── util.py
    │       │   │   ├── version.py
    │       │   │   └── versionpredicate.py
    │       │   ├── _entry_points.py
    │       │   ├── _imp.py
    │       │   ├── _importlib.py
    │       │   ├── _itertools.py
    │       │   ├── _path.py
    │       │   ├── _reqs.py
    │       │   ├── _vendor
    │       │   │   ├── __init__.py
    │       │   │   ├── importlib_metadata
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _adapters.py
    │       │   │   │   ├── _collections.py
    │       │   │   │   ├── _compat.py
    │       │   │   │   ├── _functools.py
    │       │   │   │   ├── _itertools.py
    │       │   │   │   ├── _meta.py
    │       │   │   │   └── _text.py
    │       │   │   ├── importlib_resources
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _adapters.py
    │       │   │   │   ├── _common.py
    │       │   │   │   ├── _compat.py
    │       │   │   │   ├── _itertools.py
    │       │   │   │   ├── _legacy.py
    │       │   │   │   ├── abc.py
    │       │   │   │   ├── readers.py
    │       │   │   │   └── simple.py
    │       │   │   ├── jaraco
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── context.py
    │       │   │   │   ├── functools.py
    │       │   │   │   └── text
    │       │   │   │       └── __init__.py
    │       │   │   ├── more_itertools
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── more.py
    │       │   │   │   └── recipes.py
    │       │   │   ├── ordered_set.py
    │       │   │   ├── packaging
    │       │   │   │   ├── __about__.py
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _manylinux.py
    │       │   │   │   ├── _musllinux.py
    │       │   │   │   ├── _structures.py
    │       │   │   │   ├── markers.py
    │       │   │   │   ├── requirements.py
    │       │   │   │   ├── specifiers.py
    │       │   │   │   ├── tags.py
    │       │   │   │   ├── utils.py
    │       │   │   │   └── version.py
    │       │   │   ├── pyparsing
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── actions.py
    │       │   │   │   ├── common.py
    │       │   │   │   ├── core.py
    │       │   │   │   ├── diagram
    │       │   │   │   │   └── __init__.py
    │       │   │   │   ├── exceptions.py
    │       │   │   │   ├── helpers.py
    │       │   │   │   ├── results.py
    │       │   │   │   ├── testing.py
    │       │   │   │   ├── unicode.py
    │       │   │   │   └── util.py
    │       │   │   ├── tomli
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── _parser.py
    │       │   │   │   ├── _re.py
    │       │   │   │   └── _types.py
    │       │   │   ├── typing_extensions.py
    │       │   │   └── zipp.py
    │       │   ├── archive_util.py
    │       │   ├── build_meta.py
    │       │   ├── cli-32.exe
    │       │   ├── cli-64.exe
    │       │   ├── cli-arm64.exe
    │       │   ├── cli.exe
    │       │   ├── command
    │       │   │   ├── __init__.py
    │       │   │   ├── alias.py
    │       │   │   ├── bdist_egg.py
    │       │   │   ├── bdist_rpm.py
    │       │   │   ├── build.py
    │       │   │   ├── build_clib.py
    │       │   │   ├── build_ext.py
    │       │   │   ├── build_py.py
    │       │   │   ├── develop.py
    │       │   │   ├── dist_info.py
    │       │   │   ├── easy_install.py
    │       │   │   ├── editable_wheel.py
    │       │   │   ├── egg_info.py
    │       │   │   ├── install.py
    │       │   │   ├── install_egg_info.py
    │       │   │   ├── install_lib.py
    │       │   │   ├── install_scripts.py
    │       │   │   ├── launcher manifest.xml
    │       │   │   ├── py36compat.py
    │       │   │   ├── register.py
    │       │   │   ├── rotate.py
    │       │   │   ├── saveopts.py
    │       │   │   ├── sdist.py
    │       │   │   ├── setopt.py
    │       │   │   ├── test.py
    │       │   │   ├── upload.py
    │       │   │   └── upload_docs.py
    │       │   ├── config
    │       │   │   ├── __init__.py
    │       │   │   ├── _apply_pyprojecttoml.py
    │       │   │   ├── _validate_pyproject
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── error_reporting.py
    │       │   │   │   ├── extra_validations.py
    │       │   │   │   ├── fastjsonschema_exceptions.py
    │       │   │   │   ├── fastjsonschema_validations.py
    │       │   │   │   └── formats.py
    │       │   │   ├── expand.py
    │       │   │   ├── pyprojecttoml.py
    │       │   │   └── setupcfg.py
    │       │   ├── dep_util.py
    │       │   ├── depends.py
    │       │   ├── discovery.py
    │       │   ├── dist.py
    │       │   ├── errors.py
    │       │   ├── extension.py
    │       │   ├── extern
    │       │   │   └── __init__.py
    │       │   ├── glob.py
    │       │   ├── gui-32.exe
    │       │   ├── gui-64.exe
    │       │   ├── gui-arm64.exe
    │       │   ├── gui.exe
    │       │   ├── installer.py
    │       │   ├── launch.py
    │       │   ├── logging.py
    │       │   ├── monkey.py
    │       │   ├── msvc.py
    │       │   ├── namespaces.py
    │       │   ├── package_index.py
    │       │   ├── py34compat.py
    │       │   ├── sandbox.py
    │       │   ├── script (dev).tmpl
    │       │   ├── script.tmpl
    │       │   ├── unicode_utils.py
    │       │   ├── version.py
    │       │   ├── wheel.py
    │       │   └── windows_support.py
    │       ├── setuptools-65.5.1.dist-info
    │       │   ├── INSTALLER
    │       │   ├── LICENSE
    │       │   ├── METADATA
    │       │   ├── RECORD
    │       │   ├── WHEEL
    │       │   ├── entry_points.txt
    │       │   └── top_level.txt
    │       ├── setuptools-65.5.1.virtualenv
    │       ├── wheel
    │       │   ├── __init__.py
    │       │   ├── __main__.py
    │       │   ├── _setuptools_logging.py
    │       │   ├── bdist_wheel.py
    │       │   ├── cli
    │       │   │   ├── __init__.py
    │       │   │   ├── convert.py
    │       │   │   ├── pack.py
    │       │   │   └── unpack.py
    │       │   ├── macosx_libfile.py
    │       │   ├── metadata.py
    │       │   ├── util.py
    │       │   ├── vendored
    │       │   │   ├── __init__.py
    │       │   │   └── packaging
    │       │   │       ├── __init__.py
    │       │   │       ├── _manylinux.py
    │       │   │       ├── _musllinux.py
    │       │   │       └── tags.py
    │       │   └── wheelfile.py
    │       ├── wheel-0.38.4.dist-info
    │       │   ├── INSTALLER
    │       │   ├── LICENSE.txt
    │       │   ├── METADATA
    │       │   ├── RECORD
    │       │   ├── WHEEL
    │       │   ├── entry_points.txt
    │       │   └── top_level.txt
    │       └── wheel-0.38.4.virtualenv
    ├── Scripts
    │   ├── activate
    │   ├── activate.bat
    │   ├── activate.fish
    │   ├── activate.nu
    │   ├── activate.ps1
    │   ├── activate_this.py
    │   ├── deactivate.bat
    │   ├── deactivate.nu
    │   ├── pip-3.9.exe
    │   ├── pip.exe
    │   ├── pip3.9.exe
    │   ├── pip3.exe
    │   ├── pydoc.bat
    │   ├── python.exe
    │   ├── pythonw.exe
    │   ├── wheel-3.9.exe
    │   ├── wheel.exe
    │   ├── wheel3.9.exe
    │   └── wheel3.exe
    └── pyvenv.cfg
```

---
