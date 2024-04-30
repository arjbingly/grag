Configuration
===============

GRAG gives the user an option to use a config file, in the form of a ``config.ini``.
The use of a config file streamlines the process of passing arguments to the various components in the code.

File Resolution
****************
GRAG takes the closest ``config.ini`` to the file you run. This enables users to have multiple config files per project,
if they require or just a single config file at the root.

Config Format
***************
``config.ini`` files use the `INI file format <https://en.wikipedia.org/wiki/INI_file>`_.

Each section in the config file is the name of the module and each key value pair is the arguments required by the
class in the module.

Example Config File
*******************

::

    [llm]
    model_name : Llama-2-13b-chat
    quantization : Q5_K_M
    pipeline : llama_cpp
    device_map : auto
    task : text-generation
    max_new_tokens : 1024
    temperature : 0.1
    n_batch : 1024
    n_ctx : 6000
    n_gpu_layers : -1
    std_out : True
    base_dir : ${root:root_path}/models
    
    [chroma_client]
    host : localhost
    port : 8000
    collection_name : arxiv
    embedding_type : instructor-embedding
    embedding_model : hkunlp/instructor-xl
    
    [deeplake_client]
    collection_name : arxiv
    embedding_type : instructor-embedding
    embedding_model : hkunlp/instructor-xl
    store_path : ${data:data_path}/vectordb
    
    [text_splitter]
    chunk_size : 5000
    chunk_overlap : 400
    
    [multivec_retriever]
    store_path : ${data:data_path}/doc_store
    namespace : 8c9040b0b5cd4d7cbc2e737da1b24ebf
    id_key : doc_id
    top_k : 3
    
    [parse_pdf]
    single_text_out : True
    strategy : hi_res
    infer_table_structure : True
    extract_images : True
    image_output_dir : None
    add_captions_to_text : True
    add_captions_to_blocks : True
    table_as_html : True
    
    [data]
    data_path : ${root:root_path}/data
    
    [env]
    env_path : ${root:root_path}/.env
    
    [quantize]
    llama_cpp_path : ${root:root_path}
    
    [root]
    root_path : /home/ubuntu/Capstone_5
