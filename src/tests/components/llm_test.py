from typing import Text

import pytest
from grag.components.llm import LLM

llama_models = [
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "Mixtral-8x7B-Instruct-v0.1",
    "gemma-7b-it",
]
hf_models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    # 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    "google/gemma-7b-it",
]
cpp_quantization = ["Q5_K_M", "Q5_K_M", "Q4_K_M", "f16"]
hf_quantization = ["Q8", "Q4", "Q4"]  # , 'Q4']
params = [(model, quant) for model, quant in zip(hf_models, hf_quantization)]


@pytest.mark.parametrize("hf_models, quantization", params)
def test_hf_web_pipe(hf_models, quantization):
    llm_ = LLM(quantization=quantization, model_name=hf_models, pipeline="hf")
    model = llm_.load_model(is_local=False)
    response = model.invoke("Who are you?")
    assert isinstance(response, Text)
    del model


params = [(model, quant) for model, quant in zip(llama_models, cpp_quantization)]


@pytest.mark.parametrize("model_name, quantization", params)
def test_llamacpp_pipe(model_name, quantization):
    llm_ = LLM(quantization=quantization, model_name=model_name, pipeline="llama_cpp")
    model = llm_.load_model()
    response = model.invoke("Who are you?")
    assert isinstance(response, Text)
    del model
