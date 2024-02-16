from dotenv import load_dotenv
from pathlib import Path
import os
from huggingface_hub import login
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.utils import LocalTokenNotFoundError
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from .config import llm_conf

print("CUDA: ", torch.cuda.is_available())


class LLM:
    def __init__(self,
                 model_name=llm_conf["model_name"],
                 device_map=llm_conf["device_map"],
                 task=llm_conf["task"],
                 max_new_tokens=llm_conf["max_new_tokens"],
                 temperature=llm_conf["temperature"],
                 n_batch=llm_conf["n_batch_gpu_cpp"],
                 n_ctx=llm_conf["n_ctx_cpp"],
                 n_gpu_layers=llm_conf["n_gpu_layers_cpp"],
                 ):
        self.base_dir = Path(__file__).resolve().parents[2]
        self._model_name = model_name
        self.device_map = device_map
        self.task = task
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_path(self):
        return str(
            self.base_dir / 'models' / self.model_name / f'ggml-model-{llm_conf["quantization"]}.gguf')

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    def hf_pipeline(self, is_local=False):
        if is_local:
            hf_model = self.model_path
        else:
            hf_model = self.model_name

        try:
            # Try to load the model without passing the token
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            model = AutoModelForCausalLM.from_pretrained(hf_model,
                                                         device_map=self.device_map,
                                                         torch_dtype=torch.float16, )
        except LocalTokenNotFoundError:
            # If loading fails due to an auth token error, then load the token and retry
            load_dotenv()
            auth_token = os.getenv("AUTH_TOKEN")
            if not auth_token:
                raise ValueError("Authentication token not provided.")
            tokenizer = AutoTokenizer.from_pretrained(hf_model, token=True)
            model = AutoModelForCausalLM.from_pretrained(hf_model,
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

    def llama_cpp(self):
        # https://stackoverflow.com/a/77734908/13808323
        llm = LlamaCpp(
            model_path=self.model_path,
            # max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_ctx=self.n_ctx,
            callbacks=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        return llm

    def load_model(self,
                   model_name=None,
                   pipeline='llama_cpp',
                   is_local=True):
        if model_name is not None:
            self.model_name = model_name

        match pipeline:
            case 'llama_cpp':
                return self.llama_cpp()
            case 'hf':
                return self.hf_pipeline(is_local=is_local)
