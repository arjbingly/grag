import json
import os
import textwrap
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


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


def find_config_path(current_path: Path) -> Path:
    """
    Finds the path of the 'config.ini' file by traversing up the directory tree from the current path.

    This function starts at the current path and moves up the directory tree until it finds a file named 'config.ini'.
    If 'config.ini' is not found by the time the root of the directory tree is reached, a FileNotFoundError is raised.

    Args:
        current_path (Path): The starting point for the search, typically the location of the script being executed.

    Returns:
        Path: The path to the found 'config.ini' file.

    Raises:
        FileNotFoundError: If 'config.ini' cannot be found in any of the parent directories.
    """
    config_path = Path('src/config.ini')
    while not (current_path / config_path).exists():
        current_path = current_path.parent
        if current_path == current_path.parent:
            raise FileNotFoundError(f"config.ini not found in {config_path}.")
    return current_path / config_path


def get_config(path=None) -> ConfigParser:
    """
    Retrieves and parses the configuration settings from the 'config.ini' file.

    This function locates the 'config.ini' file by calling `find_config_path` using the script's current location.
    It initializes a `ConfigParser` object to read the configuration settings from the located 'config.ini' file.

    Returns:
        ConfigParser: A parser object containing the configuration settings from 'config.ini'.
    """
    # Assuming this script is somewhere inside your project directory
    if path is None:
        script_location = Path(__file__).resolve()
        if os.environ.get('CONFIG_PATH'):
            config_path = os.environ.get('CONFIG_PATH')
        else:
            config_path = find_config_path(script_location)
            os.environ['CONFIG_PATH'] = str(config_path)
    else:
        config_path = path
    print(f"Loaded config from {config_path}.")
    # Initialize parser and read config
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_path)

    return config
