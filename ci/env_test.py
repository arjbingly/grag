import os

from grag.components.utils import get_config

get_config(load_env=True)

print(os.environ['HF_TOKEN'])
