"""Utils functions.

This module provides:

- stuff_docs: concats langchain documents into string

- load_prompt: loads json prompt to langchain prompt

- find_config_path: finds the path of the 'config.ini' file by traversing up the directory tree from the current path.

- get_config: retrieves and parses the configuration settings from the 'config.ini' file.
"""

import os
from collections import defaultdict
from configparser import ConfigParser, ExtendedInterpolation
from functools import wraps
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document


def stuff_docs(docs: List[Document]) -> str:
    r"""Concatenates langchain documents into a string using '\n\n' seperator.

    Args:
        docs: List of langchain_core.documents.Document

    Returns:
        string of document page content joined by '\n\n'
    """
    return "\n\n".join([doc.page_content for doc in docs])


def find_config_path(current_path: Path):
    """Finds the path of the 'config.ini' file by traversing up the directory tree from the current path.

    This function starts at the current path and moves up the directory tree until it finds a file named 'config.ini'.
    If 'config.ini' is not found by the time the root of the directory tree is reached, a FileNotFoundError is raised.

    Args:
        current_path (Path): The starting point for the search, typically the location of the script being executed.

    Returns:
        Path: None or the path to the found 'config.ini' file.
    """
    config_path = Path("config.ini")
    while not (current_path / config_path).exists():
        current_path = current_path.parent
        if current_path == current_path.parent:
            # raise FileNotFoundError(f"config.ini not found in {config_path}.")
            return None
    return current_path / config_path


def get_config(load_env=False):
    """Retrieves and parses the configuration settings from the 'config.ini' file.

    This function locates the 'config.ini' file by calling `find_config_path` using the script's current location.
    It initializes a `ConfigParser` object to read the configuration settings from the located 'config.ini' file.
    Optionally, it can also load environment variables from a `.env` file specified in the config.

    Args:
        load_env (bool): If True, load environment variables from the path specified in the 'config.ini'. Defaults to False.

    Returns:
        ConfigParser: A parser object containing the configuration settings from 'config.ini', or a defaultdict
                      with None if the file is not found or an empty dict{dict{}}.
    """
    config_path_ = os.environ.get("CONFIG_PATH")
    if config_path_:
        config_path = Path(config_path_)
    else:
        script_location = Path('.').resolve()
        config_path = find_config_path(script_location)
        if config_path is not None:
            os.environ["CONFIG_PATH"] = str(config_path)

    # Initialize parser and read config
    if config_path:
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(config_path)
        print(f"Loaded config from {config_path}.")
        # Load .env 
        if load_env:
            env_path = Path(config['env']['env_path'])
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment variables from {env_path}")
        return config
    else:
        return defaultdict(lambda: defaultdict(lambda: None))


def configure_args(cls):
    """Decorator to configure class instantiation arguments from a 'config.ini' file, based on the class's module name.

    This function reads configuration specific to a class's module from 'config.ini', then uses it to override or
    provide defaults for keyword arguments passed during class instantiation.

    Args:
        cls (class): The class whose instantiation is to be configured.

    Returns:
        function: A wrapped class constructor that uses modified arguments based on the configuration.

    Raises:
        TypeError: If there is a mismatch in provided arguments and class constructor requirements.
    """
    module_namespace = cls.__module__.split('.')[-1]

    config = get_config()[module_namespace]

    @wraps(cls)
    def wrapper(*args, **kwargs):
        new_kwargs = {**config, **kwargs}
        try:
            return cls(*args, **new_kwargs)
        except TypeError as e:
            raise TypeError(f"{e}, or create a config.ini file. ") from e

    return wrapper
