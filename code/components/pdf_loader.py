#%%
from langchain_community.document_loaders import UnstructuredPDFLoader

def load_split_PDF(pdf_path, mode="single", strategy="hi_res", splitter=None):
    """
    Load and split the text data from a PDF file.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        mode (str): Loading mode, "single" to get a single langchain document or "elements" for elements split (such as Title or Narrative Text).
        strategy (str): Strategy for loading, "hi_res" for high resolution or "fast" for quicker loading.
        splitter: Splitter instance to use for text splitting. Defaults to None, which uses RecursiveCharacterTextSplitter.
    
    Returns:
        pdf_doc: The split Documents from the PDF.
    """
    try:
        # Initiating PDF loader
        pdf_loader = UnstructuredPDFLoader(pdf_path, mode=mode, strategy=strategy)
        # Load and split the data
        pdf_doc = pdf_loader.load_and_split(splitter)
        return pdf_doc
    
    except Exception as e:
        # Handle or log the exception
        print(f"An error occurred: {e}")
        return None
#%%
# test cases
load_split_PDF("C:\College\DSCI CAPSTONE\FL.1942.10.pdf", mode="single")

