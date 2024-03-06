# RAG Evaluation

## Data Sources

We have used data from [KG-RAG-dataset](https://github.com/docugami/KG-RAG-datasets/tree/main) for evaluation.
They provide carefully annotated datasets for RAG evaluation.

This repository contains various datasets for advanced RAG over a multiple documents. We created these since we noticed
that existing eval datasets were not adequately reflecting RAG use cases that we see in production. Specifically, they
were doing Q&A over a single (or just a few) docs when in reality customers often need to RAG over larger sets of
documents.
QnA over multiple documents, more than just a few
Use more realistic long-form documents that are similar to documents customers use, not just standard academic examples
Include questions of varying degree of difficulty, including:
Single-Doc, Single-Chunk RAG: Questions where the answer can be found in a contiguous region (text or table chunk) of a
single doc. To correctly answer, the RAG system needs to retrieve the correct chunk and pass it to the LLM context. For
example: What did Microsoft report as its net cash from operating activities in the Q3 2022 10-Q?
Single-Doc, Multi-Chunk RAG: Questions where the answer can be found in multiple non-contiguous regions (text or table
chunks) of a single doc. To correctly answer, the RAG system needs to retrieve multiple correct chunks from a single doc
which can be challenging for certain types of questions. For example: For Amazon's Q1 2023, how does the share
repurchase information in the financial statements correlate with the equity section in the management discussion?
Multi-Doc RAG: Questions where the answer can be found in multiple non-contiguous regions (text or table chunks) across
multiple docs. To correctly answer, the RAG system needs to retrieve multiple correct chunks from multiple docs. For
example: How has Apple's revenue from iPhone sales fluctuated across quarters?
m
