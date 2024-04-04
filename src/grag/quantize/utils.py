"""Utility functions for quantization."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from grag.components.utils import get_config
from huggingface_hub import snapshot_download

config = get_config()


def get_llamacpp_repo(root_path: Union[str, Path]) -> subprocess.CompletedProcess:
    """Clones or pulls the llama.cpp repository into the specified root path.

    Args:
        root_path: The root directory where the llama.cpp repository will be cloned or updated.

    Returns:
        A subprocess.CompletedProcess instance containing the result of the git operation.
    """
    if os.path.exists(f"{root_path}/llama.cpp"):
        print(f"Repo exists at: {root_path}/llama.cpp")
        res = subprocess.run(
            ["git", "-C", f"{root_path}/llama.cpp", "pull"],
            check=True,
            capture_output=True,
        )
    else:
        res = subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/ggerganov/llama.cpp.git",
                f"{root_path}/llama.cpp",
            ],
            check=True,
            capture_output=True,
        )

    return res


def building_llamacpp(root_path: Union[str, Path]) -> None:
    """Attempts to build the llama.cpp project using make or cmake.

    Args:
        root_path (str): The root directory where the llama.cpp project is located.
    """
    os.chdir(f"{root_path}/llama.cpp/")
    try:
        subprocess.run(["which", "make"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["make", "LLAMA_CUDA=1"], check=True)
        print("Llama.cpp build successful.")
    except subprocess.CalledProcessError:
        try:
            subprocess.run(["which", "cmake"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["mkdir", "build"], check=True)
            subprocess.run(
                [
                    "cd",
                    "build",
                    "&&",
                    "cmake",
                    "..",
                    "-DLLAMA_CUDA=ON",
                    "&&",
                    "cmake",
                    "--build",
                    ".",
                    "--config",
                    "Release",
                ],
                shell=True,
                check=True,
            )
            print("Llama.cpp build successful.")
        except subprocess.CalledProcessError:
            print("Unable to build, cannot find make or cmake.")
    finally:
        os.chdir(
            Path(__file__).parent
        )  # Assuming you want to return to the root path after operation


def fetch_model_repo(repo_id: str, root_path: Union[str, Path]) -> None:
    """Download model from huggingface.co/models.

    Args:
        repo_id (str): Repository ID of the model to download.
        root_path (str): The root path where the model should be downloaded or copied.
    """
    local_dir = f"{root_path}/llama.cpp/models/{repo_id.split('/')[1]}"
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks="auto",
        resume_download=True,
    )
    print(f"Model downloaded in {local_dir}")


def quantize_model(
    model_dir_path: Union[str, Path],
    quantization: str,
    root_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Quantizes a specified model using a given quantization level.

    Args:
        output_dir (str, Path, optional): Directory to save quantized model. Defaults to None
        model_dir_path (str, Path): The directory path of the model to be quantized.
        quantization (str): The quantization level to apply.
        root_path (str, Path): The root directory path of the project.
    """
    os.chdir(f"{root_path}/llama.cpp/")
    model_dir_path = Path(model_dir_path)
    if output_dir is None:
        output_dir = config["llm"]["base_dir"]

    output_dir = Path(output_dir) / model_dir_path.name
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(["python3", "convert.py", f"{model_dir_path}/"], check=True)
    model_file = model_dir_path / "ggml-model-f32.gguf"
    quantized_model_file = output_dir / f"ggml-model-{quantization}.gguf"
    subprocess.run(
        ["./quantize", str(model_file), str(quantized_model_file), quantization],
        check=True,
    )
    print(f"Quantized model present at {output_dir}")
    os.chdir(Path(__file__).parent)  # Return to the root path after operation
