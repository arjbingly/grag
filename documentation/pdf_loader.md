## PDF Loader Documentation

### 1. Function Documentation

`load_split_PDF(pdf_path, mode="single", strategy="hi_res", splitter=None)`

- Parameters:
  - pdf_path (str): Path to the PDF file.
  - mode (str): Loading mode, "single" to get a single langchain document or "elements" for elements split (such as Title or Narrative Text).
  - strategy (str): Strategy for loading, "hi_res" for high resolution or "fast" for quicker loading.
  - splitter: Splitter instance to use for text splitting. Defaults to None, which uses RecursiveCharacterTextSplitter.
- Returns:
  - pdf_doc: The split Documents from the PDF.

### 2. Implementation

To implement locally, import `from langchain_community.document_loaders import UnstructuredPDFLoader`. Install `tesseract` and `poppler` as needed.

### 3. UnstructuredPDFLoader Documentation

From (https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html#)[Langchain Documentation]

#### Overview

Langchain class; loads and extracts data from PDF files using Unstructured.

#### Methods

`__init__(file_path: Union[str, List[str]], mode: str = 'single', **unstructured_kwargs: Any)`
Initializes the PDF loader.

- Parameters:
  - mode (str): Load mode, either "single" for single langchain Document object or "elements" for . Default is "single".
  - strategy (str): Loading strategy, either "hi_res" for high resolution or "fast" for quicker but less accuracy. Default is "hi_res".
  - load(self, pdf_path): loads the specified PDF file.

`load_and_split(text_splitter: Optional[TextSplitter] = None)`
Load Documents and split into chunks. Chunks are returned as Documents.

- Parameters:
  - text_splitter – TextSplitter instance to use for splitting documents. Defaults to RecursiveCharacterTextSplitter.
- Returns: List of Documents.
