import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


def get_llamacpp_repo(root_path: str) -> None:
    """Clones or pulls the llama.cpp repository into the specified root path.

    Args:
        root_path (str): The root directory where the llama.cpp repository will be cloned or updated.
    """
    if os.path.exists(f"{root_path}/llama.cpp"):
        print(f"Repo exists at: {root_path}/llama.cpp")
        res = subprocess.run([f"cd {root_path}/llama.cpp && git pull"], check=True, shell=True, capture_output=True)
    else:

        subprocess.run(
            [f"cd {root_path} && git clone https://github.com/ggerganov/llama.cpp.git"],
            check=True, shell=True)


def building_llama(root_path: str) -> None:
    """Attempts to build the llama.cpp project using make or cmake.

    Args:
        root_path (str): The root directory where the llama.cpp project is located.
    """
    os.chdir(f"{root_path}/llama.cpp/")
    try:
        subprocess.run(['which', 'make'], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['make', 'LLAMA_CUBLAS=1'], check=True)
        print('Llama.cpp build successful.')
    except subprocess.CalledProcessError:
        try:
            subprocess.run(['which', 'cmake'], check=True, stdout=subprocess.DEVNULL)
            subprocess.run(['mkdir', 'build'], check=True)
            subprocess.run(
                ['cd', 'build', '&&', 'cmake', '..', '-DLLAMA_CUBLAS=ON', '&&', 'cmake', '--build', '.', '--config',
                 'Release'], shell=True, check=True)
            print('Llama.cpp build successful.')
        except subprocess.CalledProcessError:
            print("Unable to build, cannot find make or cmake.")
    finally:
        os.chdir(Path(__file__).parent)  # Assuming you want to return to the root path after operation


def fetch_model_repo(repo_id: str, root_path: str) -> None:
    """Download model from huggingface.co/models.

    Args:
        repo_id (str): Repository ID of the model to download.
        root_path (str): The root path where the model should be downloaded or copied.
    """
    local_dir = f"{root_path}/llama.cpp/model/{repo_id.split('/')[1]}"
    os.mkdir(local_dir)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded in {local_dir}")


def quantize_model(model_dir_path: str, quantization: str, root_path: str) -> None:
    """Quantizes a specified model using a given quantization level.

    Args:
        model_dir_path (str): The directory path of the model to be quantized.
        quantization (str): The quantization level to apply.
        root_path (str): The root directory path of the project.
    """
    os.chdir(f"{root_path}/llama.cpp/")
    subprocess.run(["python3", "convert.py", f"models/{model_dir_path}/"], check=True)
    model_file = f"models/{model_dir_path}/ggml-model-f16.gguf"
    quantized_model_file = f"models/{model_dir_path.split('/')[-1]}/ggml-model-{quantization}.gguf"
    subprocess.run(["llm_quantize", model_file, quantized_model_file, quantization], check=True)
    print(f"Quantized model present at {root_path}/llama.cpp/{quantized_model_file}")
    os.chdir(Path(__file__).parent)  # Return to the root path after operation
