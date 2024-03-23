from grag.components.utils import get_config
from grag.quantize.utils import (
    building_llama,
    fetch_model_repo,
    get_llamacpp_repo,
    quantize_model,
)

config = get_config()
root_path = config['quantize']['llama_cpp_path']

user_input = input(
    "Enter the path to the llama_cpp cloned repo, or where you'd like to clone it. Press Enter to use the default config path: ").strip()

if user_input != "":
    root_path = user_input

res = get_llamacpp_repo(root_path)

if "Already up to date." in res.stdout:
    print("Repository is already up to date. Skipping build.")
else:
    print("Updates found. Starting build...")
    building_llama(root_path)

response = input("Do you want us to download the model? (y/n) [Enter for yes]: ").strip().lower()
if response == "n":
    print("Please copy the model folder to 'llama.cpp/models/' folder.")
    _ = input("Enter if you have already copied the model:")
    model_dir = input("Enter the model directory name: ")
elif response == "y" or response == "":
    repo_id = input('Please enter the repo_id for the model (you can check on https://huggingface.co/models): ').strip()
    fetch_model_repo(repo_id, root_path)
    model_dir = repo_id.split('/')[1]

quantization = input(
    "Enter quantization, recommended - Q5_K_M or Q4_K_M for more check https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19 : ")
quantize_model(model_dir, quantization, root_path)
