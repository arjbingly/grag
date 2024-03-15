## Model Quantization

This directory contains tools to quantize the models supported by `llama.cpp`.

### 1. Steps to start with llama.cpp:

- Clone [this](https://github.com/ggerganov/llama.cpp) repository as below  
  `git clone https://github.com/ggerganov/llama.cpp.git`
- Change directory to `llama.cpp` using  
  `cd llama.cpp`
- Now make sure you have CUDA installed (check using `nvcc --version`) and
  go [here](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#cublas) and follow the steps.

Note: While inferencing if model is not utilizing GPU check the `BLAS=1` in the outputs and if it is not then try reinstalling using `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
` or follow
solution [here](https://stackoverflow.com/questions/76963311/llama-cpp-python-not-using-nvidia-gpu-cuda).

### 2. Downloading models to quantize:

- Make sure you are in `models` directory in `llama.cpp` for this step.
- Go to [HuggingFace](https://huggingface.co/models) and search for the models, make sure the model is listed
  in [description](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#description) of `llama.cpp`.
- Follow the steps to download models on HuggingFace.

Note: If you are using gated models (like Llama-2) request for access from HuggingFace and Meta, steps will be in model
repository.
After you have finished downloading models check if you have all the files downloaded properly using checksum.

### 3. Steps to quantize:

- Make sure you are in `llm_quantize` directory. Now check that there is a `quantize.py` file in there.
- Run below command to quantize the model  
  `python quantize.py <model_directory_name> <quantization_method>`
- Example: `python quantize Llama-2-7B-chat Q5_K_M`  
  This will create a `ggml-model-Q5_K_M.gguf` that has everything in it for you to load model.

If there is some issue during quantization
check [here](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#description).
To know more on which quantization method to use
look [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF). [These](https://github.com/ggerganov/llama.cpp/blob/89febfed9322c8849520dc63c93ee4f5fd72556e/examples/quantize/quantize.cpp#L19)
are the supported quantization methods.

Once you have your quantized model move it to `models/` directory in root folder (if it doesn't exist `mkdir models`).
In this directory Make a file with model name (e.g. `mkdir Llama-2-13b-chat`) and put the quantized model there.
