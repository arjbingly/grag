LLMs
=====

GRAG offers two ways to run LLMs locally:

1. LlamaCPP
2. HuggingFace

To run LLMs using HuggingFace
#############################
This is the easiest way to get started, but does not offer as much
flexibility.
If using a config file (*config.ini*), just change the `model_name` to
to the HuggingFace repo id. *Note that if the models are gated, make sure to
provide an auth token*.

To run LLMs using LlamaCPP
#############################
LlamaCPP requires models in the form of `.gguf` file. You can either download these model files online,
or **quantize** the model yourself following the instructions below.

How to quantize models
***********************
To quantize the model, run:
  ``python -m grag.quantize.quantize``

After running the above command, user will be prompted with the following:

1. **Path** where the user wants to clone the `llama.cpp` repo. You can find the repository, `llama.cpp <https://github.com/ggerganov/llama.cpp>`_.

2.  Input the **model path**:

* If user wants to download a model from `HuggingFace <https://huggingface.co/models>`_, the user should provide the repository path or URL from HuggingFace.

* If the user has the model downloaded locally, then user will be instructed to copy the model and input the name of the model directory.

3. The user will be asked where to put the quantized model otherwise it will go in the directory where you downloaded model repository.

4. Finally, the user will be prompted to enter **quantization** settings (recommended Q5_K_M or Q4_K_M, etc.). For more details, check `llama.cpp/examples/quantize/quantize.cpp <https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19>`_.

5. Optionally, user can inference the quantized model with the next prompt. This inference will be on CPU so it takes time if model is large one.

Note: Windows users have to use WSL, and follow linux guidelines for quantizing models.
