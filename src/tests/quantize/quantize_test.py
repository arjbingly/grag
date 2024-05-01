import os
import shutil
from pathlib import Path

import pytest
from grag.quantize.utils import (
    get_llamacpp_repo,
    get_asset_download_url,
    download_release_asset,
    fetch_model_repo,

)

root_path = Path(__file__).parent / "test_quantization"
if os.path.exists(root_path):
    shutil.rmtree(root_path)
os.makedirs(root_path, exist_ok=True)

asset_pattern_list = ['-macos-x64', '-macos-arm64', '-win-arm64-x64', '-win-arm64-x64', '-ubuntu-x64']


def test_get_llamacpp_repo():
    get_llamacpp_repo(root_path)
    repo_path = root_path / "llama.cpp" / ".git"
    assert os.path.exists(repo_path)


@pytest.mark.parametrize("asset_pattern", asset_pattern_list)
def test_get_asset_download_url(asset_pattern):
    url = get_asset_download_url(asset_pattern, 'ggerganov', 'llama.cpp')
    response = requests.get(url, stream=True)
    assert response.status_code == 200


@pytest.mark.parametrize("asset_pattern", asset_pattern_list)
def test_download_release_asset(asset_pattern):
    url = get_asset_download_url(asset_pattern, 'ggerganov', 'llama.cpp')
    response = requests.get(url, stream=True)
    download_release_asset(response, root_path)
    assert os.path.exists(root_path / 'build' / 'bin' / 'quantize')
    assert os.path.exists(root_path / 'build' / 'bin' / 'main')


def test_fetch_model_repo():
    fetch_model_repo("meta-llama/Llama-2-7b-chat", root_path / 'models')
    model_dir_path = root_path / "models" / "Llama-2-7b-chat"
    assert os.path.exists(model_dir_path)


def test_quantize_model():
    model_dir_path = root_path / "llama.cpp" / "models" / "Llama-2-7b-chat"
    quantize_model(
        model_dir_path, "Q3_K_M", root_path, output_dir=model_dir_path.parent
    )
    gguf_file_path = model_dir_path / "ggml-model-Q3_K_M.gguf"
    assert os.path.exists(gguf_file_path)
