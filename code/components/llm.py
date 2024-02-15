from dotenv import load_dotenv
from pathlib import Path
import os
from huggingface_hub import login
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from config import llm_conf
print("CUDA: ", torch.cuda.is_available())


class LLM:
    def __init__(self):
        self.llm_model = llm_conf["model_path"]
        base_dir = Path(__file__).resolve().parent.parent
        self.model_path = base_dir / 'models' / self.llm_model / f'ggml-model-{llm_conf["quantization"]}.gguf'
        self.device_map = llm_conf["device_map"]
        self.task = llm_conf["task"]
        self.max_new_tokens = llm_conf["max_new_tokens"]
        self.temperature = llm_conf["temperature"]
        self.n_batch = llm_conf["n_batch_gpu_cpp"]
        self.n_ctx = llm_conf["n_ctx_cpp"]
        self.n_gpu_layers = llm_conf["n_gpu_layers_cpp"]
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    def hf_pipeline(self):
        try:
            load_dotenv()
            # Access the environment variable
            auth_token = os.getenv("AUTH_TOKEN")
            if not auth_token:
                raise ValueError("Authentication token not provided.")
            login(token=auth_token)

            tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                      token=True)

            model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                         device_map=self.device_map,
                                                         torch_dtype=torch.float16,
                                                         token=True)

            pipe = pipeline(self.task,
                            model=model,
                            tokenizer=tokenizer,
                            torch_dtype=torch.bfloat16,
                            device_map=self.device_map,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True,
                            top_k=30,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id
                            )
            llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': self.temperature})

            return llm
        except ValueError as e:
            print(e)
            return None

    def llama_cpp(self):
        # https://stackoverflow.com/a/77734908/13808323
        llm = LlamaCpp(
            model_path=self.model_path,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_ctx=self.n_ctx,
            callbacks=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        return llm
