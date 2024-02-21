## Folder Structure
```
src
├── Readme.md  
├── __init__.py  
├── components -> Contains all reusable modularised code   
│   ├── __init__.py  
│   ├── chroma_client.py   
│   ├── config.py  
│   ├── embedding.py  
│   ├── llm.py  
│   ├── multivec_retriever.py  
│   ├── parse_pdf.py  
│   ├── text_splitter.py
│   └── utils.py
├── scripts -> Contains scripts to run, reset, etc. 
│   ├── reset_chroma.sh
│   ├── reset_store.sh
│   └── run_chroma.sh
├── tests -> Contains test files for components 
│   ├── chroma_add_test.py
│   ├── chroma_async_test.py
│   ├── embedding_test.py
│   ├── llm_test.py
│   ├── multivec_retriever_test.py
│   └── parse_pdf_test.py
└── utils 
    └── txt_data_ingest.py
```