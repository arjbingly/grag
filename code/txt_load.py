from langchain.document_loaders import TextLoader

class TextDocumentLoader:
    def __init__(self, text_location):
        self.text_location = text_location

    def load(self):
        try:
            loader = TextLoader(self.text_location)
            document = loader.load()
            return document
        except Exception as e:
            print(f"Error loading text document: {e}")
            return None