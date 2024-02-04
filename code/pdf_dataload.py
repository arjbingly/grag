from langchain_community.document_loaders import PyPDFLoader
class PDFLoader:
    def __init__(self, pdf_location):
        self.pdf_location = pdf_location

    def load_and_split(self):
        pages = []
        try:
            loader = PyPDFLoader(self.pdf_location)
            total_pages = len(loader.pages)

            for page_num in range(total_pages):
                page = loader.pages[page_num]
                # You can customize how you want to store or process each page
                pages.append(page.extract_text())

            return pages

        except Exception as e:
            print(f"Error loading and splitting PDF: {e}")
            return None
