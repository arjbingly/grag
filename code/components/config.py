
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
from uuid import UUID

chroma_conf = {
    'host' : 'localhost',
    'port' : 8000,
    'collection_name' : "arxiv",
    # 'embedding_type' : 'sentence-transformers',
    # 'embedding_model' : "all-mpnet-base-v2",
    'embedding_type' : 'instructor-embedding',
    'embedding_model' : 'hkunlp/instructor-xl',
    'store_path' : "data/vectordb",
    'allow_reset': True
}

text_splitter_conf = {
    'chunk_size': 5000,
    'chunk_overlap': 400,
}

multivec_retriever_conf = {
    'store_path': 'data/docs',
    'namespace': UUID('8c9040b0-b5cd-4d7c-bc2e-737da1b24ebf'),
    # 'namespace': '8c9040b0b5cd4d7cbc2e737da1b24ebf',
    'id_key': 'doc_id',

}
