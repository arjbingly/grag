"""Utility functions for quantization."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Union

import requests
from git import Repo
from grag.components.utils import get_config
from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import GatedRepoError

config = get_config()


def get_llamacpp_repo(repo_url: str = 'https://github.com/ggerganov/llama.cpp.git',
                      destination_folder: Union[str, Path] = './grag-quantize') -> None:
    """Clones a GitHub repository to a specified local directory or updates it if it already exists. The directory is created if it does not exist. If the repository is already cloned, it pulls updates.

    Args:
        repo_url: The URL of the repository to clone.
        destination_folder: The local path where the repository should be cloned or updated.

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
        asset_name_pattern: Substring to match in the asset's name.
        user: GitHub username or organization of the repository.
        repo: Repository name.

    Returns:
        The download URL of the matching asset, or None if no match is found.
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


def download_release_asset(download_url: str, root_quantize: Union[Path, str] = './grag-quantize') -> None:
    """Downloads a file from a given URL and saves it to a specified path. It also attempts to extract the file if it is a ZIP archive.

    Args:
        download_url: The URL of the file to download.
        root_quantize: Path where the file will be saved.

    Returns:
        None
    """
    root_quantize = Path(root_quantize)
    root_quantize.mkdir(parents=True, exist_ok=True)
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        with open(root_quantize / 'llamacpp_release.zip', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded successfully to {root_quantize}")
        with zipfile.ZipFile(root_quantize / 'llamacpp_release.zip', 'r') as zip_ref:
            # Extract all the contents into the destination directory
            zip_ref.extractall(root_quantize)
            print(f"Files extracted to {root_quantize}")
    else:
        print(f"Failed to download file: {response.status_code}")


def repo_id_resolver(repo_url: str) -> str:
    """Resolves the HuggingFace repository ID given a full URL to a model or dataset page.

    This function parses a HuggingFace URL to extract the repository ID, which typically
    consists of a user or organization name followed by the repository name. If the URL
    does not start with the expected HuggingFace URL prefix, it returns the input URL unchanged.

    Args:
        repo_url: The full URL string pointing to a specific HuggingFace repository.

    Returns:
        The repository ID in the format 'username/repository_name' if the URL is valid,
        otherwise returns the original URL.

    Examples:
        Input: "https://huggingface.co/gpt2/models"
        Output: "gpt2/models"

        Input: "https://huggingface.co/facebook/bart-large"
        Output: "facebook/bart-large"

        Input: "some_other_url"
        Output: "some_other_url"
    """
    if repo_url.startswith('https://huggingface'):
        repo_url = repo_url.rstrip(' ')
        repo_url = repo_url.lstrip(' ')
        repo_url = repo_url.rstrip('/')
        repo_lst = repo_url.split('/')
        return f'{repo_lst[-2]}/{repo_lst[-1]}'
    else:
        return repo_url


def fetch_model_repo(repo_id: str, model_path: Union[str, Path] = './grag-quantize/models') -> Union[str, Path]:
    """Downloads a model from huggingface.co/models to a specified directory.

    Args:
        repo_id: Repository ID of the model to download (e.g., 'huggingface/gpt2').
        model_path: The local directory where the model should be downloaded.

    Returns:
        The path to the directory where the model is downloaded.
    """
    model_path = Path(model_path)
    local_dir = model_path / f"{repo_id.split('/')[1]}"
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks="auto",
            resume_download=True,
        )
    except GatedRepoError:
        print(
            "This model comes under gated repository. You must be authenticated to download the model. For more: https://huggingface.co/docs/hub/en/models-gated")
        resp = input(
            "You will be redirected to hugginface-cli to login. If you don't have token checkout above link or else paste the token when prompted. [To exit, enter 'n']: ")
        if resp.lower() == "n":
            print("User exited.")
            exit(0)
        elif resp == "":
            login()
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks="auto",
                resume_download=True,
            )
        else:
            raise ValueError('Invalid response received.')
    print(f"Model downloaded in {local_dir}")
    return local_dir


def quantize_model(
    model_dir_path: Union[str, Path],
    quantization: str,
    root_quantize: Union[str, Path] = './grag-quantize',  # path with both build and llamacpp
    output_dir: Optional[Union[Path, str]] = None,
) -> Tuple[Path, Path]:
    """Quantizes a specified model using a given quantization level and saves it to an optional directory. If the output directory is not specified, it defaults to a subdirectory under the provided model directory. The function also handles specific exceptions during the conversion process and ensures the creation of the necessary directories.

    Args:
        model_dir_path: The directory path of the model to be quantized. This path must exist and contain the model files.
        quantization: The quantization level to apply (e.g., 'f32', 'f16'). This affects the precision and size of the model.
        root_quantize: The root directory containing the quantization tools and scripts. This directory should have the necessary binary files and scripts for the quantization process.
        output_dir: Optional directory to save the quantized model. If not specified, the function uses a default directory based on the model directory path.

    Returns:
        Tuple[Path, Path]: Returns a tuple containing the path to the root of the quantization tools and the path to the quantized model file.
        
    Raises:
        PermissionError: If the function lacks permissions to execute the quantization binaries, it will attempt to modify permissions and retry.
        TypeError: If there are issues with the provided model directory or quantization parameters.
    """
    model_dir_path = Path(model_dir_path).resolve()
    if output_dir == '' or output_dir is None:
        try:
            output_dir = Path(config["llm"]["base_dir"])
        except (KeyError, TypeError):
            output_dir = model_dir_path
    else:
        output_dir = Path(output_dir)

    output_dir = output_dir / model_dir_path.name if output_dir.name != model_dir_path.name else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()

    root_quantize = Path(root_quantize).resolve()
    os.chdir(root_quantize / 'llama.cpp')
    convert_script_path = os.path.join(root_quantize, 'llama.cpp')
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

    quantized_model_file = output_dir / f"ggml-model-{quantization}.gguf"
    if not os.path.exists(quantized_model_file):
        converted_model_file = output_dir / "ggml-model-f32.gguf"
        binary_path = root_quantize / 'build' / 'bin' / 'quantize'
        cmd = [str(binary_path), str(converted_model_file), str(quantized_model_file), quantization]

        try:
            subprocess.run(cmd, check=True)
        except PermissionError:
            os.chmod(binary_path, 0o777)
            subprocess.run(cmd, check=True)
        print(f"Quantized model present at {output_dir}")
    else:
        print("Quantized model already exists for given quantization, skipping...")
    os.chdir(Path(__file__).parent)  # Return to the root path after operation

    return root_quantize, quantized_model_file


def inference_quantized_model(root_quantize: Union[str, Path],
                              quantized_model_file: Union[str, Path]) -> subprocess.CompletedProcess:
    """Runs inference using a quantized model binary.

    Args:
        root_quantize: The root directory containing the compiled inference executable.
        quantized_model_file: The file path to the quantized model to use for inference.

    Returns:
        The subprocess.CompletedProcess object containing the inference execution result.
    """
    root_quantize = Path(root_quantize)
    main_path = root_quantize / 'build' / 'bin' / 'main'
    run_cmd = [str(main_path), '-m', str(quantized_model_file), '-ngl', '-1']
    try:
        res = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
    except PermissionError:
        os.chmod(main_path, 0o777)
        res = subprocess.run(run_cmd, check=True, text=True, capture_output=True)
    print('Inference successfull for this quantized model.')
    return res
