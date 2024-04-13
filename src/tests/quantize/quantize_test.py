import os
import shutil
from pathlib import Path

from grag.quantize.utils import (
    building_llamacpp,
    fetch_model_repo,
    get_llamacpp_repo,
    quantize_model,
)

root_path = Path(__file__).parent / "test_data"
if os.path.exists(root_path):
    shutil.rmtree(root_path)
os.makedirs(root_path, exist_ok=True)


def test_get_llamacpp_repo():
    get_llamacpp_repo(root_path)
    repo_path = root_path / "llama.cpp" / ".git"
    assert os.path.exists(repo_path)


def test_build_llamacpp():
    building_llamacpp(root_path)
    bin_path = root_path / "llama.cpp" / "quantize"
    assert os.path.exists(bin_path)


def test_fetch_model_repo():
    fetch_model_repo("meta-llama/Llama-2-7b-chat", root_path)
    model_dir_path = root_path / "llama.cpp" / "models" / "Llama-2-7b-chat"
    assert os.path.exists(model_dir_path)


def test_quantize_model():
    model_dir_path = root_path / "llama.cpp" / "models" / "Llama-2-7b-chat"
    quantize_model(
        model_dir_path, "Q3_K_M", root_path, output_dir=model_dir_path.parent
    )
    gguf_file_path = model_dir_path / "ggml-model-Q3_K_M.gguf"
    assert os.path.exists(gguf_file_path)
