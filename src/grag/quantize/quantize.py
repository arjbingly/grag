"""Interactive file for quantizing models."""

import platform
import sys
from pathlib import Path

from grag.components.utils import get_config
from grag.quantize.utils import (
    download_release_asset,
    fetch_model_repo,
    get_asset_download_url,
    get_llamacpp_repo,
    inference_quantized_model,
    quantize_model,
    repo_id_resolver,
)

config = get_config()

if __name__ == "__main__":
    user_input = input(
        "Enter the path which you want to download all the source files. Press Enter to use the default path: ").strip()

    if user_input == "":
        try:
            root_path = Path(config["quantize"]["llama_cpp_path"])
            print(f'Using {root_path} from config.ini')
        except (KeyError, TypeError):
            root_path = Path('./grag-quantize')
            print(f'Using {root_path}, default.')
    else:
        root_path = Path(user_input)

    get_llamacpp_repo(destination_folder=root_path)
    os_name = str(platform.system()).lower()
    architecture = str(platform.machine()).lower()
    asset_name_pattern = 'bin'
    match os_name, architecture:
        case ('darwin', 'x86_64'):
            asset_name_pattern += '-macos-x64'
        case ('darwin', 'arm64'):
            asset_name_pattern += '-macos-arm64'
        case ('windows', 'x86_64'):
            asset_name_pattern += '-win-arm64-x64'
        case ('windows', 'arm64'):
            asset_name_pattern += '-win-arm64-x64'
        case ('windows', 'amd64'):
            asset_name_pattern += '-win-arm64-x64'
        case ('linux', 'x86_64'):
            asset_name_pattern += '-ubuntu-x64'
        case _:
            raise ValueError(f"{os_name=}, {architecture=} is not supported by llama.cpp releases.")

    download_url = get_asset_download_url(asset_name_pattern)
    if download_url:
        download_release_asset(download_url, root_path)

    response = input("Do you want us to download the model? (yes[y]/no[n]) [Enter for yes]: ").strip().lower()
    if response == '':
        response = 'yes'
    if response.lower()[0] == "n":
        model_dir = Path(input("Enter path to the model directory: "))
    elif response.lower()[0] == "y":
        repo_id = input(
            "Please enter the repo_id or the url for the model (you can check on https://huggingface.co/models): "
        ).strip()
        if repo_id == "":
            raise ValueError("Repo ID you entered was empty. Please enter the repo_id for the model.")
        repo_id = repo_id_resolver(repo_id)
        model_dir = fetch_model_repo(repo_id, root_path / 'models')
    else:
        raise ValueError("Please enter either 'yes', 'y' or 'no', 'n'.")

    sys.stdin.flush()

    output_dir = input(
        f"Enter path where you want to save the quantized model, else the following path will be used [{model_dir}]: ").strip()
    quantization = input(
        "Enter quantization, recommended - Q5_K_M or Q4_K_M for more check https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19 : "
    ).strip()

    target_path, quantized_model_file = quantize_model(model_dir, quantization, root_path, output_dir)

    inference = input(
        "Do you want to inference the quantized model to check if quantization is successful? Warning: It takes time as it inferences on CPU. (y/n) [Enter for yes]: ").strip().lower()
    if response == '':
        response = 'yes'
    if response.lower()[0] == "y":
        inference_quantized_model(target_path, quantized_model_file)
    else:
        print("Model quantized, but not tested.")
