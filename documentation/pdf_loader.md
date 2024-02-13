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

### 4. Notes

Slow for scanned PDF files. Accuracy is decent, but dependent on scan/image quality.
