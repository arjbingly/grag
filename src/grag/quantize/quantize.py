"""Interactive file for quantizing models."""

import platform
from pathlib import Path

from grag.components.utils import get_config
from grag.quantize.utils import (
    download_release_asset,
    fetch_model_repo,
    get_asset_download_url,
    get_llamacpp_repo,
    inference_quantized_model,
    quantize_model,
)

config = get_config()

if __name__ == "__main__":
    user_input = input(
        "Enter the path which you want to download all the source files. Press Enter to use the default path: ").strip()

    if user_input == "":
        try:
            root_path = Path(config["quantize"]["llama_cpp_path"])
        except KeyError:
            root_path = Path('./grag-quantize')
    else:
        root_path = Path(user_input)

    get_llamacpp_repo(destination_folder=root_path)
    os_name = platform.system()
    architecture = platform.machine()
    asset_name_pattern = 'bin'
    match os_name, architecture:
        case ('Darwin', 'x86_64'):
            asset_name_pattern += '-macos-x64'
        case ('Darwin', 'arm64'):
            asset_name_pattern += '-macos-arm64'
        case ('Windows', 'x86_64'):
            asset_name_pattern += '-win-arm64-x64'
        case ('Windows', 'arm64'):
            asset_name_pattern += '-win-arm64-x64'
        case ('Linux', 'x86_64'):
            asset_name_pattern += '-ubuntu-x64'
        case _:
            raise ValueError(f"{os_name=}, {architecture=} is not supported by llama.cpp releases.")

    download_url = get_asset_download_url(asset_name_pattern)
    if download_url:
        download_release_asset(download_url, root_path)

    response = input("Do you want us to download the model? (y/n) [Enter for yes]: ").strip().lower()
    if response == "n":
        model_dir = Path(input("Enter path to the model directory: "))
    elif response == "y" or response == "":
        repo_id = input(
            "Please enter the repo_id for the model (you can check on https://huggingface.co/models): "
        ).strip()
        if repo_id == "":
            raise ValueError("Repo ID you entered was empty. Please enter the repo_id for the model.")
        model_dir = fetch_model_repo(repo_id, root_path / 'models')

    quantization = input(
        "Enter quantization, recommended - Q5_K_M or Q4_K_M for more check https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19 : "
    ).strip()
    output_dir = input(
        f"Enter path where you want to save the quantized model, else the following path will be used [{model_dir}]: ").strip()

    target_path, quantized_model_file = quantize_model(model_dir, quantization, root_path, output_dir)

    inference = input(
        "Do you want to inference the quantized model to check if quantization is successful? Warning: It takes time as it inferences on CPU. (y/n) [Enter for yes]: ").strip().lower()
    inference_quantized_model(target_path, quantized_model_file) if inference == "y" or inference == "" else None
