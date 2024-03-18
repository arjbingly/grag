from typing import Text

import pytest
from grag.components.llm import LLM

#
# models_to_test = ['Llama-2-7b-chat',
#                   'Llama-2-13b-chat',
#                   'Mixtral-8x7B-Instruct-v0.1',
#                   'gemma-7b-it']
#
# pipeline_list = ['llama_cpp', 'hf']

llama_models = ['Llama-2-7b-chat',
                'Llama-2-13b-chat',
                'Mixtral-8x7B-Instruct-v0.1',
                'gemma-7b-it']
hf_models = ['meta-llama/Llama-2-7b-chat',
             'meta-llama/Llama-2-13b-chat',
             'mistralai/Mixtral-8x7B-Instruct-v0.1',
             'google/gemma-7b-it']
cpp_quantization = ['Q5_K_M', 'Q5_K_M', 'Q4_K_M', 'f16']
hf_quantization = ['Q8', 'Q4', 'Q4', 'Q4']

params = [(model, 'llama_cpp', quant) for model, quant in zip(llama_models, cpp_quantization)]
params.extend([(model, 'hf', quant) for model, quant in zip(llama_models, cpp_quantization)])


@pytest.mark.parametrize("model_name, pipeline, quantization", params)
def test_model(model_name, pipeline, quantization):
    llm_ = LLM(quantization=quantization, model_name=model_name, pipeline=pipeline)
    model = llm_.load_model()
    response = model.invoke("Who are you?")
    assert isinstance(response, Text)
    del model
