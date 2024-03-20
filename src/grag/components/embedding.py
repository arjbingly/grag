from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


class Embedding:
    """
    A class for vector embeddings.
    Supports:
        huggingface sentence transformers -> model_type = 'sentence-transformers'
        huggingface instructor embeddings -> model_type = 'instructor-embedding'

    Attributes:
        embedding_type: embedding model type, refer above for supported types
        embedding_model: huggingface model name
        embedding_function: langchain embedding type
    """

    def __init__(self, embedding_type: str, embedding_model: str):
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        match self.embedding_type:
            case "sentence-transformers":
                self.embedding_function = SentenceTransformerEmbeddings(
                    model_name=self.embedding_model
                )
            case "instructor-embedding":
                self.embedding_instruction = "Represent the document for retrival"
                self.embedding_function = HuggingFaceInstructEmbeddings(
                    model_name=self.embedding_model
                )
                self.embedding_function.embed_instruction = self.embedding_instruction
            case _:
                raise Exception("embedding_type is invalid")
