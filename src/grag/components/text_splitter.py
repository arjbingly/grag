"""Class for splitting/chunking text.

This module provides:
- TextSplitter
"""

from typing import Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .utils import get_config

text_splitter_conf = get_config()["text_splitter"]


# %%
class TextSplitter:
    """Class for recursively chunking text, it prioritizes '/n/n then '/n' and so on.

    Attributes:
        chunk_size: maximum size of chunk
        chunk_overlap: chunk overlap size
    """

    def __init__(
        self,
        chunk_size: Union[int, str] = text_splitter_conf["chunk_size"],
        chunk_overlap: Union[int, str] = text_splitter_conf["chunk_overlap"],
    ):
        """Initialize TextSplitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            length_function=len,
            is_separator_regex=False,
        )
        """Initialize TextSplitter using chunk_size and chunk_overlap"""
