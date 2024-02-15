
llm_conf = {
    'model_path': '../../../llama.cpp/models/Llama-2-70b-chat/ggml-model-Q4_K_M.gguf',  # 'meta-llama/Llama-2-70b-chat-hf',
    'quantization': 'Q5_K_M',
    'device_map': 'auto',
    'task': 'text-generation',
    'max_new_tokens': 1024,
    'temperature': 0,
    'n_batch_gpu_cpp': 1024,
    'n_ctx_cpp': 1536,
    'n_gpu_layers_cpp': 12,  # The number of layers to put on the GPU.

}
