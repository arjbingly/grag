import os
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path


# Function to find the root directory
def find_project_root(current_path):
    # Traverse up until you find the config.ini file
    while not (current_path / 'CONFIG_Run' / 'config' / 'config.ini').exists():
        current_path = current_path.parent
        if current_path == current_path.parent:
            raise FileNotFoundError("config.ini not found in any parent directories.")
    return current_path


# Function to get config object
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


# Usage
config = get_config()
print(config['instance']['user'])  # Example of accessing a config value
