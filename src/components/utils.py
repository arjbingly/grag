import os
import json
import textwrap
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.components.multivec_retriever import Retriever


def stuff_docs(docs: List[Document]) -> str:
    """
    Args:
        docs: List of langchain_core.documents.Document

    Returns:
        string of document page content joined by '\n\n'
    """
    return '\n\n'.join([doc.page_content for doc in docs])


def reformat_text_with_line_breaks(input_text, max_width=110):
    """
    Reformat the given text to ensure each line does not exceed a specific width,
    preserving existing line breaks.

    Args:
    input_text (str): The text to be reformatted.
    max_width (int): The maximum width of each line.

    Returns:
    str: The reformatted text with preserved line breaks and adjusted line width.
    """
    # Divide the text into separate lines
    original_lines = input_text.split('\n')

    # Apply wrapping to each individual line
    reformatted_lines = [textwrap.fill(line, width=max_width) for line in original_lines]

    # Combine the lines back into a single text block
    reformatted_text = '\n'.join(reformatted_lines)

    return reformatted_text


def display_llm_output_and_sources(response_from_llm):
    """
    Displays the result from an LLM response and lists the sources.

    Args:
    response_from_llm (dict): The response object from an LLM which includes the result and source documents.
    """
    # Display the main result from the LLM response
    print(response_from_llm['result'])

    # Separator for clarity
    print('\nSources:')

    # Loop through each source document and print its source
    for source in response_from_llm["source_documents"]:
        print(source.metadata['source'])


def get_prompt(query):
    """
    Generates a chat prompt using a predefined template and documents retrieved based on the given query.

    Args:
        query (str): The user query to generate a prompt for.

    Returns:
        str: The generated prompt.
    """
    # Load a prompt template from a JSON file
    with open("json_path", "r") as f:
        prompt_json = json.load(f)
    prompt_template = ChatPromptTemplate.from_template(prompt_json['template'])

    # Retrieve documents related to the query
    retrieved_docs = retriever.get_chunk(query)

    # Process the retrieved documents to generate context
    context = stuff_docs(retrieved_docs)

    # Format the prompt template with the generated context and the query
    prompt = prompt_template.format(context=context, question=query)

    return prompt
