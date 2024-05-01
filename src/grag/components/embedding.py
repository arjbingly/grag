"""Class for embedding.

This module provides:

— Embedding
"""

from grag.components.utils import gen_str
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


class Embedding:
    """A class for vector embeddings.

    Supports:
        huggingface sentence transformers -> model_type = 'sentence-transformers'
        huggingface instructor embeddings -> model_type = 'instructor-embedding'

    Attributes:
        embedding_type: embedding model type, refer above for supported types
        embedding_model: huggingface model name
        embedding_function: langchain embedding type
    """

    def __init__(self, embedding_type: str, embedding_model: str):
        """Initialize the embedding with embedding_type and embedding_model."""
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        match self.embedding_type:
            case "sentence-transformers":
                self.embedding_function = SentenceTransformerEmbeddings(
                    model_name=self.embedding_model  # type: ignore
                )
            case "instructor-embedding":
                self.embedding_instruction = "Represent the document for retrival"
                self.embedding_function = HuggingFaceInstructEmbeddings(
                    model_name=self.embedding_model  # type: ignore
                )
                self.embedding_function.embed_instruction = self.embedding_instruction  # type: ignore
            case _:
                raise Exception("embedding_type is invalid")

    def __call__(self):
        """Embed the document."""
        return self.embedding_function()

    # def __str__(self):
    #     repr_string = "Embedding (\n"
    #     repr_string += f"\ttype: {self.embedding_type},\n"
    #     repr_string += f"\tmodel: {self.embedding_model},\n"
    #     repr_string += f"\tmax_seq_len: {self.embedding_function.client.max_seq_length},\n"
    #     repr_string += f"\tdevice: {self.embedding_function.client.device},\n"
    #     if self.embedding_type == "instructor-embedding":
    #         repr_string += f"\tembedding_instruction: {self.embedding_function.embed_instruction},\n"
    #     repr_string += ")"
    #     return repr_string

    def __str__(self):
        """Return a string representation of the object."""
        dict = {
            "embedding_type": self.embedding_type,
            "embedding_model": self.embedding_model,
            "max_seq_length": self.embedding_function.client.max_seq_length,
            "device": self.embedding_function.client.device,
        }
        if self.embedding_type == "sentence-transformers":
            dict["embedding_instruction"] = self.embedding_function.embedding_instruction
        return gen_str(self, dict)
