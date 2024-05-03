## Model Quantization

This module provides interactive way to quantize your model.
To quantize model:  
`python -m grag.quantize.quantize`

After running the above command user will be prompted with following:

- Path where user want to clone [llama.cpp](!https://github.com/ggerganov/llama.cpp) repo.
- If user wants us to download model from [HuggingFace](!https://huggingface.co/models) or user has model downloaded
  locally.
- For former, user will be prompted to provide repo path from HuggingFace.
- In case of later, user will be instructed to copy the model and input the name of model directory.
- Finally, user will be prompted to enter quantization (recommended Q5_K_M or Q4_K_M, etc.). Check
  more [here](!https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19).