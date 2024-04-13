LLMs
=====

GRAG offers two ways to run LLMs locally,

1. LlamaCPP
2. HuggingFace

To run LLMs using HuggingFace
#############################
This is the easiest way to get started but does not offer as much
flexibility.
If using a config file (*config.ini*), just change the `model_name` to
to the HuggingFace repo id. *Note that if the models are gated, make sure to
provide an auth token*

To run LLMs using LlamaCPP
#############################
Steps to start with llama.cpp:

1. Clone the `llama.cpp <https://github.com/ggerganov/llama.cpp>`_ repository.
  ``git clone https://github.com/ggerganov/llama.cpp.git``
2. Change directory to `llama.cpp` using `cd llama.cpp`
3. To inference using GPU, which is necessary for most models.
  * Make sure you have CUDA installed (check using ``nvcc --version``)
  * Follow steps from the `llama.cpp documentation <https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#cublas>`_.

*Note: While inferencing if model is not utilizing GPU check the `BLAS=1` in the outputs and*
*if it is not then try reinstalling using*::

    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

*or follow the solution provided by*
`this Stack Overflow post <https://stackoverflow.com/questions/76963311/llama-cpp-python-not-using-nvidia-gpu-cuda>`_

How to quantize models.
************************
