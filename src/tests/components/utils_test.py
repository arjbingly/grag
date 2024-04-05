import os

from grag.components.utils import get_config


def test_get_config():
    config = get_config(load_env=True)
    assert os.environ["HF_TOKEN"]
