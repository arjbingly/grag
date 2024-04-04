from typing import Text

import pytest
from grag.components.llm import LLM
from grag.components.utils import get_config

config = get_config(load_env=True)

llama_models = [
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "gemma-7b-it",
    "Mixtral-8x7B-Instruct-v0.1",
]
hf_models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "google/gemma-7b-it",
]
cpp_quantization = ["Q5_K_M", "Q5_K_M", "f16", "Q4_K_M"]
gpu_layers = ['-1', '-1', '18', '16']
hf_quantization = ["Q8", "Q4", "Q4"]
params = [(model, quant) for model, quant in zip(hf_models, hf_quantization)]


@pytest.mark.parametrize("hf_models, quantization", params)
def test_hf_web_pipe(hf_models, quantization):
    llm_ = LLM(quantization=quantization, model_name=hf_models, pipeline="hf")
    model = llm_.load_model(is_local=False)
    response = model.invoke("Who are you?")
    assert isinstance(response, Text)
    del model


params = [(model, gpu_layer, quant) for model, gpu_layer, quant in zip(llama_models, gpu_layers, cpp_quantization)]


@pytest.mark.parametrize("model_name, gpu_layer, quantization", params)
def test_llamacpp_pipe(model_name, gpu_layer, quantization):
    llm_ = LLM(quantization=quantization, model_name=model_name, n_gpu_layers=gpu_layer, pipeline="llama_cpp")
    model = llm_.load_model()
    response = model.invoke("Who are you?")
    assert isinstance(response, Text)
    del model
