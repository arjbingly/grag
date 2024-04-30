"""Utility functions for quantization."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Union

import requests
from git import Repo
from grag.components.utils import get_config
from huggingface_hub import snapshot_download

config = get_config()


def get_llamacpp_repo(repo_url: str = 'https://github.com/ggerganov/llama.cpp.git',
                      destination_folder: Union[str, Path] = './grag-quantize') -> None:
    """Clones a GitHub repository to a specified local directory or updates it if it already exists using GitPython.

    Args:
        repo_url (str): The URL of the repository to clone.
        destination_folder (str, Path): The local path where the repository should be cloned or updated.

    Returns:
        None
    """
    destination_folder = Path(destination_folder) / 'llama.cpp'
    destination_folder.mkdir(parents=True, exist_ok=True)
    if os.path.isdir(destination_folder) and os.path.isdir(os.path.join(destination_folder, '.git')):
        try:
            repo = Repo(destination_folder)
            origin = repo.remotes.origin
            origin.pull()
            print(f"Repository updated successfully in {destination_folder}")
        except Exception as e:
            print(f"Failed to update repository: {str(e)}")
    else:
        try:
            Repo.clone_from(repo_url, destination_folder)
            print(f"Repository cloned successfully into {destination_folder}")
        except Exception as e:
            print(f"Failed to clone repository: {str(e)}")


def get_asset_download_url(asset_name_pattern: str, user: str = 'ggerganov', repo: str = 'llama.cpp') -> Optional[str]:
    """Fetches the download URL of the first asset that matches a given name pattern in the latest release of the specified repository.

    Args:
        user (str): GitHub username or organization of the repository.
        repo (str): Repository name.
        asset_name_pattern (str): Substring to match in the asset's name.

    Returns:
        str: The download URL of the matching asset, or None if no match is found.
    """
    url = f"https://api.github.com/repos/{user}/{repo}/releases/latest"
    response = requests.get(url)
    if response.status_code == 200:
        release = response.json()
        for asset in release.get('assets', []):
            if asset_name_pattern in asset['name']:
                return asset['browser_download_url']
        print("No asset found matching the pattern.")
    else:
        print("Failed to fetch release info:", response.status_code)
    return None


def download_release_asset(download_url: str, target_path: Union[Path, str] = './grag-quantize') -> None:
    """Downloads a file from a given URL and saves it to a specified path.

    Args:
        download_url (str): The URL of the file to download.
        target_path (str, Path): Path where the file will be saved.
    """
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(target_path / 'llamacpp_release.zip', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded successfully to {target_path}")
        with zipfile.ZipFile(target_path / 'llamacpp_release.zip', 'r') as zip_ref:
            # Extract all the contents into the destination directory
            zip_ref.extractall(target_path)
            print(f"Files extracted to {target_path}")
    else:
        print(f"Failed to download file: {response.status_code}")


def fetch_model_repo(repo_id: str, model_path: Union[str, Path] = './grag-quantize/models') -> None:
    """Download model from huggingface.co/models.

    Args:
        repo_id (str): Repository ID of the model to download.
        model_path (str): The root path where the model should be downloaded or copied.
    """
    model_path = Path(model_path)
    local_dir = model_path / f"{repo_id.split('/')[1]}"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks="auto",
        resume_download=True,
    )
    print(f"Model downloaded in {local_dir}")
    return local_dir


def exec_quantize(quantized_model_file: Union[str, Path], cmd: list):
    if not os.path.exists(quantized_model_file):
        subprocess.run(cmd, check=True)
    else:
        print("Quantized model already exists for given quantization, skipping...")


def quantize_model(
    model_dir_path: Union[str, Path],
    quantization: str,
    target_path: Union[str, Path] = './grag-quantize',  # path with both bulid and llamacpp
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Quantizes a specified model using a given quantization level.

    Args:
        output_dir (str, Path, optional): Directory to save quantized model. Defaults to None
        model_dir_path (str, Path): The directory path of the model to be quantized.
        quantization (str): The quantization level to apply.
        root_path (str, Path): The root directory path of the project.
    """
    # os.chdir(f"{root_path}/llama.cpp/")
    model_dir_path = Path(model_dir_path).resolve()
    if output_dir is None:
        try:
            output_dir = config["llm"]["base_dir"]
        except KeyError:
            output_dir = Path('.')

    output_dir = Path(output_dir) / model_dir_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()

    target_path = Path(target_path).resolve()
    os.chdir(target_path / 'llama.cpp')
    convert_script_path = os.path.join(target_path, 'llama.cpp')
    sys.path.append(convert_script_path)

    from convert import main as convert

    args_list = [f'{model_dir_path}',
                 '--outfile', f'{output_dir}/ggml-model-f32.gguf']
    if not os.path.exists(f'{output_dir}/ggml-model-f32.gguf'):
        try:
            convert(args_list)
        except TypeError as e:
            if 'with BpeVocab' in str(e):
                args_list.extend(['--vocab-type', 'bpe'])
                convert(args_list)
            else:
                raise e
    else:
        print('f32 gguf file already exists, skipping conversion...')

    model_file = output_dir / "ggml-model-f32.gguf"
    quantized_model_file = output_dir / f"ggml-model-{quantization}.gguf"
    binary_path = target_path / 'build' / 'bin' / 'quantize'

    cmd = [str(binary_path), str(model_file), str(quantized_model_file), quantization]

    try:
        exec_quantize(quantized_model_file, cmd)
    except PermissionError:
        os.chmod(binary_path, 0o777)
        exec_quantize(quantized_model_file, cmd)

    print(f"Quantized model present at {output_dir}")

    os.chdir(Path(__file__).parent)  # Return to the root path after operation

    return target_path, quantized_model_file


def inference_quantized_model(target_path: Union[str, Path], quantized_model_file: Union[str, Path]):
    main_path = target_path / 'build' / 'bin' / 'main'
    run_cmd = [str(main_path), '-m', str(quantized_model_file)]
    try:
        res = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
    except PermissionError:
        os.chmod(main_path, 0o777)
        res = subprocess.run(run_cmd, check=True, text=True, capture_output=True)

    if subprocess.CalledProcessError:
        raise RuntimeError(subprocess.CalledProcessError.stderr)
    else:
        print('Inference successfull for this quantized model.')
        # print(res.stdout)
