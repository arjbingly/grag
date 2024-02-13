from langchain_community.document_loaders import UnstructuredMarkdownLoader

class MarkdownLoader:
    def __init__(self, markdown_location):
        self.markdown_location = markdown_location

    def load(self):
        try:
            loader = UnstructuredMarkdownLoader(self.markdown_location, mode="elements", strategy="fast")
            docs = loader.load()
            return docs

        except Exception as e:
            print(f"Error loading Markdown document: {e}")
            return None