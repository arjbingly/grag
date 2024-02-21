import subprocess
import sys
import os


def execute_commands(model_dir_path, quantization=None):
    if os.path.isdir('llama.cpp'):

        # Convert the model to ggml FP16 format
        subprocess.run(["python3", "llama.cpp/convert.py", f"llama.cpp/models/{model_dir_path}/"], check=True)

        # Quantize the model only if a quantization method is given
        if quantization:
            model_file = f"llama.cpp/models/{model_dir_path}/ggml-model-f16.gguf"
            quantized_model_file = f"llama.cpp/models/{model_dir_path.split('/')[-1]}/ggml-model-{quantization}.gguf"
            subprocess.run(["llama.cpp/llm_quantize", model_file, quantized_model_file, quantization], check=True)

    else:
        print("llama.cpp doesn't exist, check readme how to clone.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script.py <model_dir_name> [<quantization>]")
        sys.exit(1)
    model_dir_path = sys.argv[1]
    quantization = sys.argv[2] if len(sys.argv) == 3 else None
    execute_commands(model_dir_path, quantization)
