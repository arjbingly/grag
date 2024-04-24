# GRAG - Good RAG

![GitHub License](https://img.shields.io/github/license/arjbingly/Capstone_5)
![Linting](https://img.shields.io/github/actions/workflow/status/arjbingly/Capstone_5/sphinx-gitpg.yml?label=Docs)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/arjbingly/Capstone_5/build_linting.yml?label=Linting)
![Static Badge](https://img.shields.io/badge/Tests-passing-darggreen)
![Static Badge](https://img.shields.io/badge/docstring%20style-google-yellow)
![Static Badge](https://img.shields.io/badge/linter%20-ruff-yellow)
![Static Badge](https://img.shields.io/badge/buildstyle-hatchling-purple?labelColor=white)
![Static Badge](https://img.shields.io/badge/codestyle-pyflake-purple?labelColor=white)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues-pr/arjbingly/Capstone_5)

[![Static Badge][Documentation-badge]][Docuementation-url]
[![Static Badge][Cookbooks-badge]][Cookbooks-url]

[GRAG](https://arjbingly.github.io/Capstone_5/) is a simple python package that provides an easy end-to-end solution for implementing Retrieval Augmented Generation (RAG).

The package offers an easy way for running various LLMs locally, Thanks to LlamaCpp and also supports vector stores like Chroma and DeepLake. It also makes it easy to integrage support to any vector stores easy.

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
  - [Supported Vector Databases](#supported-vector-databases)

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

### Supported Vector Databases

**1. [Chroma](https://www.trychroma.com)**

Since Chroma is a server-client based vector database, make sure to run the server.

- To run Chroma locally, move to `src/scripts` then run `source run_chroma.sh`. This by default runs on port 8000.
- If Chroma is not run locally, change `host` and `port` under `chroma` in `src/config.ini`.

**2. [Deeplake](https://www.deeplake.ai/)**


For more information refer to [Documentation](https://arjbingly.github.io/Capstone_5/).  


[Documentation-badge]: https://img.shields.io/badge/Documentation-red.svg?style=for-the-badge
[Docuementation-url]: https://arjbingly.github.io/Capstone_5/
[Cookbooks-badge]: https://img.shields.io/badge/Cookbooks-blue?style=for-the-badge
[Cookbooks-url]: https://arjbingly.github.io/Capstone_5/auto_examples_index.html
