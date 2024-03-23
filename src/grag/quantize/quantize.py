from grag.components.utils import get_config
from grag.quantize.utils import (
    building_llama,
    fetch_model_repo,
    get_llamacpp_repo,
    quantize_model,
)

config = get_config()
root_path = config["quantize"]["llama_cpp_path"]

user_input = input(
    "Enter the path to the llama_cpp cloned repo, or where you'd like to clone it. Press Enter to use the default config path: ").strip()

<<<<<<< HEAD
if user_input != "":
    root_path = user_input
=======
def get_llamacpp_repo():
    if os.path.exists(f"{root_path}/llama.cpp"):
        subprocess.run(
            [f"cd {root_path}/llama.cpp && git pull"], check=True, shell=True
        )
    else:
        subprocess.run(
            [f"cd {root_path} && git clone https://github.com/ggerganov/llama.cpp.git"],
            check=True,
            shell=True,
        )
>>>>>>> aec73771f90cdeeaa9d4fe128fc04fb5befb786e

res = get_llamacpp_repo(root_path)

<<<<<<< HEAD
if "Already up to date." in res.stdout:
    print("Repository is already up to date. Skipping build.")
else:
    print("Updates found. Starting build...")
    building_llama(root_path)
=======
def building_llama():
    os.chdir(f"{root_path}/llama.cpp/")
    try:
        subprocess.run(["which", "make"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["make", "LLAMA_CUBLAS=1"], check=True)
        print("Llama.cpp build successfull.")
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
                    "-DLLAMA_CUBLAS=ON",
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
            print("Llama.cpp build successfull.")
        except subprocess.CalledProcessError:
            print("Unable to build, cannot find make or cmake.")
    os.chdir(original_dir)
>>>>>>> aec73771f90cdeeaa9d4fe128fc04fb5befb786e

response = input("Do you want us to download the model? (y/n) [Enter for yes]: ").strip().lower()
if response == "n":
    print("Please copy the model folder to 'llama.cpp/models/' folder.")
    _ = input("Enter if you have already copied the model:")
    model_dir = input("Enter the model directory name: ")
elif response == "y" or response == "":
    repo_id = input('Please enter the repo_id for the model (you can check on https://huggingface.co/models): ').strip()
    fetch_model_repo(repo_id, root_path)
    model_dir = repo_id.split('/')[1]

<<<<<<< HEAD
quantization = input(
    "Enter quantization, recommended - Q5_K_M or Q4_K_M for more check https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19 : ")
quantize_model(model_dir, quantization, root_path)
=======
def fetch_model_repo():
    response = (
        input("Do you want us to download the model? (yes/no) [Enter for yes]: ")
        .strip()
        .lower()
    )
    if response == "no":
        print("Please copy the model folder to 'llama.cpp/models/' folder.")
    elif response == "yes" or response == "":
        repo_id = input(
            "Please enter the repo_id for the model (you can check on https://huggingface.co/models): "
        )
        local_dir = f"{root_path}/llama.cpp/model/{repo_id.split('/')[1]}"
        os.mkdir(local_dir)
        snapshot_download(
            repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False
        )
        print(f"Model downloaded in {local_dir}")


def quantize_model(quantization):
    os.chdir(f"{root_path}/llama.cpp/")
    subprocess.run(["python3", "convert.py", f"models/{model_dir_path}/"], check=True)

    model_file = f"models/{model_dir_path}/ggml-model-f16.gguf"
    quantized_model_file = (
        f"models/{model_dir_path.split('/')[-1]}/ggml-model-{quantization}.gguf"
    )
    subprocess.run(
        ["llm_quantize", model_file, quantized_model_file, quantization], check=True
    )
    print(f"Quantized model present at {root_path}/llama.cpp/{quantized_model_file}")
    os.chdir(original_dir)


if __name__ == "__main__":
    get_llamacpp_repo()
    building_llama()
    fetch_model_repo()

    quantization = input("Enter quantization: ")
    quantize_model(quantization)
    # if len(sys.argv) < 2 or len(sys.argv) > 3:
    #     print("Usage: python script.py <model_dir_name> [<quantization>]")
    #     sys.exit(1)
    # model_dir_path = sys.argv[1]
    # quantization = sys.argv[2] if len(sys.argv) == 3 else None
    # execute_commands(model_dir_path, quantization)
>>>>>>> aec73771f90cdeeaa9d4fe128fc04fb5befb786e
