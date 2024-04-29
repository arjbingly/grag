"""Class for LLM."""

import os
from pathlib import Path
from typing import Optional, Union

import torch
from grag.components.utils import configure_args
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


@configure_args
class LLM:
    """A class for managing and utilizing large language models (LLMs).

    Attributes:
        model_name (str): Name of the model to be loaded.
        device_map (dict): Device mapping for model execution.
        task (str): The task for which the model is being used.
        max_new_tokens (int): Maximum new tokens to be generated.
        temperature (float): Sampling temperature for generation.
        n_batch (int): Number of batches for GPU CPP.
        n_ctx (int): Context size for CPP.
        n_gpu_layers (int): Number of GPU layers for CPP.
    """

    def __init__(
            self,
            model_name: str,
            device_map: str,
            task: str,
            max_new_tokens: str,
            temperature: str,
            n_batch: str,
            n_ctx: str,
            n_gpu_layers: str,
            std_out: Union[bool, str],
            base_dir: str,
            quantization: str,
            pipeline: str,
            callbacks=None,
    ):
        """Initialize the LLM class using the given parameters."""
        self.base_dir = Path(base_dir)
        self._model_name = model_name
        self.quantization = quantization
        self.pipeline = pipeline
        self.device_map = device_map
        self.task = task
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = temperature
        self.n_batch = n_batch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        if std_out:
            self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        else:
            self.callback_manager = callbacks  # type: ignore

    @property
    def model_name(self):
        """Returns the name of the model."""
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        """Returns the path to the model."""
        self._model_name = value

    @property
    def model_path(self):
        """Sets the model name."""
        return str(
            self.base_dir / self.model_name / f"ggml-model-{self.quantization}.gguf"
        )

    def hf_pipeline(self, is_local: Optional[bool] = False):
        """Loads the model using Hugging Face transformers.

        Args:
            is_local (bool): Whether to load the model from a local path.
        """
        if is_local:
            hf_model = Path(self.model_path).parent
        else:
            hf_model = self.model_name
            match self.quantization:
                case "Q8":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                case "Q4":
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                case _:
                    raise ValueError(
                        f"{self.quantization} is not a valid quantization. Non-local hf_pipeline takes only Q4 and Q8."
                    )

        try:
            # Try to load the model without passing the token
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                quantization_config=quantization_config,
                device_map=self.device_map,
                torch_dtype=torch.float16,
            )
        except OSError:  # LocalTokenNotFoundError:
            # If loading fails due to an auth token error, then load the token and retry
            # load_dotenv()
            if not os.getenv("HF_TOKEN"):
                raise ValueError("Authentication token not provided.")
            tokenizer = AutoTokenizer.from_pretrained(hf_model, token=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                quantization_config=quantization_config,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                token=True,
            )

        pipe = pipeline(
            self.task,
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        llm = HuggingFacePipeline(
            pipeline=pipe, model_kwargs={"temperature": self.temperature}
        )
        return llm

    def llama_cpp(self):
        """Loads the model using a custom CPP pipeline."""
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

    def load_model(
            self, model_name: Optional[str] = None, pipeline: Optional[str] = None, quantization: Optional[str] = None,
            is_local: Optional[bool] = None
    ):
        """Loads the model based on the specified pipeline and model name.

        Args:
            quantization (str): Quantization of the LLM model like Q5_K_M, f16, etc. Optional.
            model_name (str): The name of the model to load. Optional.
            pipeline (str): The pipeline to use for loading the model. Defaults to 'llama_cpp'.
            is_local (bool): Whether the model is loaded from a local directory. Defaults to True.
        """
        if model_name is not None:
            self.model_name = model_name
        if pipeline is not None:
            self.pipeline = pipeline
        if quantization is not None:
            self.quantization = quantization
        if is_local is None:
            is_local = False

        match self.pipeline:
            case "llama_cpp":
                return self.llama_cpp()
            case "hf":
                return self.hf_pipeline(is_local=is_local)
