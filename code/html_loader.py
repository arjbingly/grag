from langchain_community.document_loaders import UnstructuredHTMLLoader

class HTMLLoader:
    def __init__(self, html_location):
        self.html_location = html_location

    def load(self):
        try:
            loader = UnstructuredHTMLLoader(self.html_location, mode="elements", strategy="fast")
            '''use "elements" mode, the unstructured
           library will split the document into elements such as Title and NarrativeText'''
            docs = loader.load()
            return docs

        except Exception as e:
            print(f"Error loading HTML document: {e}")
            return None