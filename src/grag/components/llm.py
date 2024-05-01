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

    This class facilitates the loading and operation of large language models using different pipelines and settings.
    It supports both local and Hugging Face-based model management, with adjustable parameters for quantization,
    computational specifics, and output control.

    Attributes:
        model_name (str): Name of the model to be loaded.
        quantization (str): Quantization setting for the model, affecting performance and memory usage.
        pipeline (str): Type of pipeline ('llama_cpp' or 'hf') used for model operations.
        device_map (str): Device mapping for model execution, defaults to 'auto'.
        task (str): The task for which the model is being used, defaults to 'text-generation'.
        max_new_tokens (int): Maximum number of new tokens to be generated, defaults to 1024.
        temperature (float): Sampling temperature for generation, affecting randomness.
        n_batch (int): Number of batches for GPU CPP, impacting batch processing.
        n_ctx (int): Context size for CPP, defining the extent of context considered.
        n_gpu_layers (int): Number of GPU layers for CPP, specifying computational depth.
        std_out (bool or str): Flag or descriptor for standard output during operations.
        base_dir (str or Path): Base directory path for model files, defaults to 'models'.
        callbacks (list or None): List of callback functions for additional processing.
    """

    def __init__(
        self,
        model_name: str,
        quantization: str,
        pipeline: str,
        device_map: str = "auto",
        task: str = "text-generation",
        max_new_tokens: str = "1024",
        temperature: Union[str, int] = 0.1,
        n_batch: Union[str, int] = 1024,
        n_ctx: Union[str, int] = 6000,
        n_gpu_layers: Union[str, int] = -1,
        std_out: Union[bool, str] = True,
        base_dir: Union[str, Path] = Path("models"),
        callbacks=None,
    ):
        """Initialize the LLM class using the given parameters.

        Args:
            model_name (str): Specifies the model name.
            quantization (str): Sets the model's quantization configuration.
            pipeline (str): Determines which pipeline to use for model operations.
            device_map (str, optional): Device configuration for model deployment.
            task (str, optional): Defines the specific task or use-case of the model.
            max_new_tokens (int, optional): Limits the number of tokens generated in one operation.
            temperature (float, optional): Controls the generation randomness.
            n_batch (int, optional): Adjusts batch processing size.
            n_ctx (int, optional): Configures the context size used in model operations.
            n_gpu_layers (int, optional): Sets the depth of computation in GPU layers.
            std_out (bool or str, optional): Manages standard output settings.
            base_dir (str or Path, optional): Specifies the directory for storing model files.
            callbacks (list, optional): Provides custom callback functions for runtime.
        """
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
        self,
        model_name: Optional[str] = None,
        pipeline: Optional[str] = None,
        quantization: Optional[str] = None,
        is_local: Optional[bool] = None,
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
