import os
import json
import textwrap
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


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


def load_prompt(json_file: str | os.PathLike, return_input_vars=False):
    """
    Loads a prompt template from json file and returns a langchain ChatPromptTemplate

    Args:
        json_file: path to the prompt template json file.
        return_input_vars: if true returns a list of expected input variables for the prompt.

    Returns:
        langchain_core.prompts.ChatPromptTemplate (and a list of input vars if return_input_vars is True)

    """
    with open(f"{json_file}", "r") as f:
        prompt_json = json.load(f)
    prompt_template = ChatPromptTemplate.from_template(prompt_json['template'])

    input_vars = prompt_json['input_variables']

    return (prompt_template, input_vars) if return_input_vars else prompt_template


def find_project_root(current_path):
    # Traverse up until you find the config.ini file
    while not (current_path / 'CONFIG_Run' / 'config' / 'config.ini').exists():
        current_path = current_path.parent
        if current_path == current_path.parent:
            raise FileNotFoundError("config.ini not found in any parent directories.")
    return current_path


def get_config():
    # Assuming this script is somewhere inside your project directory
    script_location = Path(__file__).resolve()
    root_directory = find_project_root(script_location)

    # Path to your config file
    config_path = root_directory / 'CONFIG_Run' / 'config' / 'config.ini'
    print(config_path)
    # Initialize parser and read config
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_path)

    return config
