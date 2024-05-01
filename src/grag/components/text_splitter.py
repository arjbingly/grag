"""Class for splitting/chunking text.

This module provides:

— TextSplitter
"""

from typing import Union

from grag.components.utils import configure_args, get_str
from langchain.text_splitter import RecursiveCharacterTextSplitter


@configure_args
class TextSplitter:
    """Class for recursively chunking text, it prioritizes '/n/n then '/n' and so on.

    Attributes:
        chunk_size: maximum size of chunk
        chunk_overlap: chunk overlap size
    """

    def __init__(
            self, chunk_size: Union[int, str] = 2000, chunk_overlap: Union[int, str] = 400
    ):
        """Initialize TextSplitter using chunk_size and chunk_overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            length_function=len,
            is_separator_regex=False,
        )

    def __str__(self):
        """Return string representation of the object."""
        dict = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        return get_str(self, dict)
