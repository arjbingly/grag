import os
import shutil
from pathlib import Path

import pytest
import requests
from grag.quantize.utils import (
    get_llamacpp_repo,
    get_asset_download_url,
    download_release_asset,
    repo_id_resolver,
    fetch_model_repo,
    quantize_model,
    inference_quantized_model,
)

root_path = Path(__file__).parent / "test_quantization"
if os.path.exists(root_path):
    shutil.rmtree(root_path)
os.makedirs(root_path, exist_ok=True)

repo_id = 'meta-llama/Llama-2-7b-chat'
repo_url = 'https://huggingface.co/meta-llama/Llama-2-7b-chat'
model = 'Llama-2-7b-chat'
quantization = 'Q2_K'
asset_pattern_list = ['-macos-x64', '-macos-arm64', '-win-arm64-x64', '-win-arm64-x64', '-ubuntu-x64']


def test_get_llamacpp_repo():
    get_llamacpp_repo(destination_folder=root_path)
    repo_path = root_path / "llama.cpp" / ".git"
    assert os.path.exists(repo_path)


@pytest.mark.parametrize("asset_pattern", asset_pattern_list)
def test_get_asset_download_url(asset_pattern):
    url = get_asset_download_url(asset_pattern, 'ggerganov', 'llama.cpp')
    response = requests.get(url, stream=True)
    assert response.status_code == 200


def test_download_release_asset():
    asset_pattern = '-ubuntu-x64'
    url = get_asset_download_url(asset_pattern, 'ggerganov', 'llama.cpp')
    download_release_asset(url, root_path)
    assert os.path.exists(root_path / 'build' / 'bin' / 'quantize')
    assert os.path.exists(root_path / 'build' / 'bin' / 'main')


def test_repo_id_resolver():
    repo_id_ = repo_id_resolver(repo_url)
    assert repo_id == repo_id_


def test_fetch_model_repo():
    local_dir = fetch_model_repo(repo_id, root_path / 'models')
    assert os.path.exists(local_dir)


def test_quantize_model():
    model_dir_path = root_path / "models" / model
    # output_dir = root_path / "models" / "Llama-2-7b-chat"
    quantize_model(model_dir_path, quantization, root_path, model_dir_path)
    gguf_file_path = model_dir_path / f"ggml-model-{quantization}.gguf"
    assert os.path.exists(gguf_file_path)


def test_inference_quantized_model():
    quantized_model_file = root_path / 'models' / model / f'ggml-model-{quantization}.gguf'
    res = inference_quantized_model(root_path, quantized_model_file)
    assert isinstance(res.stdout, str)
