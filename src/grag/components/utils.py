"""Utils functions.

This module provides:

— stuff_docs: concats langchain documents into string

— load_prompt: loads json prompt to langchain prompt

— find_config_path: finds the path of the 'config.ini' file by traversing up the directory tree from the current path.

— get_config: retrieves and parses the configuration settings from the 'config.ini' file.

— configure_args: a decorator to configure class instantiation arguments from a 'config.ini' file.
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
    If 'config.ini' is not found by the time the root of the directory tree is reached, None is returned.

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
    If a config file cannot be read, a default dictionary is returned.

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
        script_location = Path(".").resolve()
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
            env_path = Path(config["env"]["env_path"])
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
    module_namespace = cls.__module__.split(".")[-1]

    config = get_config()[module_namespace]

    @wraps(cls)
    def wrapper(*args, **kwargs):
        new_kwargs = {**config, **kwargs}
        try:
            return cls(*args, **new_kwargs)
        except TypeError as e:
            raise TypeError(f"{e}, or create a config.ini file. ") from e

    return wrapper


def intend_obj_str(obj):
    r"""Formats the string representation of an object by indenting each line.

    This function takes the `__str__` output of an object, and indents every
    new line to improve readability, particularly useful for nested object 
    structures.

    Args:
        obj (object): The object whose string representation will be indented.

    Returns:
        str: The indented string representation of `obj`.

    Examples:
        class MyClass:
            def __str__(self):
                return "MyClass:\nattribute: value"

        my_object = MyClass()
        print(intend_obj_str(my_object))
        # Output:
        # MyClass:
        #    attribute: value
    """
    return obj.__str__().replace('\n', '\n\t')


def gen_str(obj, dict):
    """Generates a formatted string representation for an object based on a dictionary of its attributes.

    This function constructs a string that represents an object in a more readable form, where each attribute
    from the provided dictionary is presented in a line, prefixed by its key. String values are added directly, 
    while objects are intended using `intend_obj_str`.

    Args:
        obj (object): The object for which the string representation is being generated.
        dict (dict): A dictionary where keys are attribute names and values are attribute values of `obj`.

    Returns:
        str: A formatted string representation of `obj` showing its class name and attributes.

    Raises:
        TypeError: If a value in `dict` is neither a string nor an object with a valid `__str__` method.

    Examples:
        class MyClass:
            def __init__(self, name, details):
                self.name = name
                self.details = details

            def __str__(self):
                return "MyClass details"

        my_object = MyClass("Example", {"key": "value"})
        attributes = {'name': 'Example', 'details': my_object}
        print(gen_str(my_object, attributes))
        # Output:
        # MyClass(
        #     name: Example,
        #     details: MyClass details
        # )
    """
    str_string = f"{type(obj).__name__}(\n"
    for key, value in dict.items():
        if isinstance(value, str):
            str_string += f"\t{key}: {value},\n"
        elif isinstance(value, object):
            str_string += f"\t{key}: {intend_obj_str(value)},\n"
        else:
            raise TypeError(f"{value}, is neither a string or object with __str__ ") from value
    str_string += ")"
    return str_string
