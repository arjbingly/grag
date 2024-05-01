"""Runnable file for creating a default config.ini file."""

import shutil
from pathlib import Path
from typing import Union

import grag.resources
from importlib_resources import files


def create_config(path: Union[str, Path] = ".") -> None:
    """Create a configuration file if it doesn't exist.

    This function checks for the existence of a 'config.ini' file at the given path.
    If the file does not exist, it copies a default configuration file from the package's
    resources to the specified location. If the file already exists, it notifies the user
    and does not overwrite the existing file.

    Args:
        path (Union[str, Path]): The directory path where the 'config.ini' should be
                                 located. If not specified, defaults to the current
                                 directory ('.').

    Returns:
        None

    Raises:
        FileNotFoundError: If the default configuration file does not exist.
        PermissionError: If the process does not have permission to write to the specified
                         directory.
    """
    default_config_path = files(grag.resources).joinpath("default_config.ini")
    path = Path(path) / "config.ini"
    path = path.resolve()
    if path.exists():
        print("Config file already exists")
    else:
        shutil.copyfile(default_config_path, path, follow_symlinks=True)
        print(f"Created config file at {path}")


if __name__ == "__main__":
    create_config()
