## Model Quantization

This module provides an interactive way to quantize your model.
To quantize model, run:  
`python -m grag.quantize.quantize`

After running the above command, user will be prompted with the following:

- Path where user wants to clone [llama.cpp](!https://github.com/ggerganov/llama.cpp) repo
- If user wants us to download model from [HuggingFace](!https://huggingface.co/models) or user has model downloaded
  locally
  - For the former, user will be prompted to provide repo path from HuggingFace
  - For the latter, user will be instructed to copy the model and input the name of model directory
- Finally, user will be prompted to enter quantization (recommended Q5_K_M or Q4_K_M, etc.). For more details, check [here](!https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19).
