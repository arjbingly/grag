"""Classes for prompts.

This module provides:

- Prompt - for generic prompts

- FewShotPrompt - for few-shot prompts
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator

Example = Dict[str, Any]

SUPPORTED_TASKS = ["QA"]
SUPPORTED_DOC_CHAINS = ["stuff", "refine"]


class Prompt(BaseModel):
    """A class for generic prompts.

    Attributes:
        name (str): The prompt name (Optional, defaults to "custom_prompt")
        llm_type (str): The type of llm, llama2, etc (Optional, defaults to "None")
        task (str): The task (Optional, defaults to QA)
        source (str): The source of the prompt (Optional, defaults to "NoSource")
        doc_chain (str): The doc chain for the prompt ("stuff", "refine") (Optional, defaults to "stuff")
        language (str): The language of the prompt (Optional, defaults to "en")
        filepath (str): The filepath of the prompt (Optional)
        input_keys (List[str]): The input keys for the prompt
    template (str): The template for the prompt
    """

    name: str = Field(default="custom_prompt")
    llm_type: str = Field(default="None")
    task: str = Field(default="QA")
    source: str = Field(default="NoSource")
    doc_chain: str = Field(default="stuff")
    language: str = "en"
    filepath: Optional[str] = Field(default=None, exclude=True)
    input_keys: List[str]
    template: str
    prompt: Optional[PromptTemplate] = Field(exclude=True, repr=False, default=None)

    @field_validator("input_keys")
    @classmethod
    def validate_input_keys(cls, v) -> List[str]:
        """Validate the input_keys field."""
        if v is None or v == []:
            raise ValueError("input_keys cannot be empty")
        return v

    @field_validator("doc_chain")
    @classmethod
    def validate_doc_chain(cls, v: str) -> str:
        """Validate the doc_chain field."""
        if v not in SUPPORTED_DOC_CHAINS:
            raise ValueError(
                f"The provided doc_chain, {v} is not supported, supported doc_chains are {SUPPORTED_DOC_CHAINS}"
            )
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        """Validate the task field."""
        if v not in SUPPORTED_TASKS:
            raise ValueError(
                f"The provided task, {v} is not supported, supported tasks are {SUPPORTED_TASKS}"
            )
        return v

    # @model_validator(mode='after')
    # def load_template(self):
    #     self.prompt = ChatPromptTemplate.from_template(self.template)
    def __init__(self, **kwargs):
        """Initialize the prompt."""
        super().__init__(**kwargs)
        self.prompt = PromptTemplate(
            input_variables=self.input_keys, template=self.template
        )

    def save(
        self, filepath: Union[Path, str, None], overwrite=False
    ) -> Union[None, ValueError]:
        """Saves the prompt class into a json file."""
        dump = self.model_dump_json(indent=2, exclude_defaults=True, exclude_none=True)
        if filepath is None:
            filepath = f"{self.name}.json"
        if overwrite:
            if self.filepath is None:
                return ValueError("filepath does not exist in instance")
            filepath = self.filepath
        with open(filepath, "w") as f:
            f.write(dump)
        return None

    @classmethod
    def load(cls, filepath: Union[Path, str]):
        """Loads a json file and returns a Prompt class."""
        with open(f"{filepath}", "r") as f:
            prompt_json = json.load(f)
        _prompt = cls(**prompt_json)
        _prompt.filepath = str(filepath)
        return _prompt

    def format(self, **kwargs) -> str:
        """Formats the prompt with provided keys and returns a string."""
        if self.prompt is not None:
            return self.prompt.format(**kwargs)
        raise ValueError("Prompt is not defined.")


class FewShotPrompt(Prompt):
    """A class for generic prompts.

    Attributes:
        name (str): The prompt name (Optional, defaults to "custom_prompt") (Parent Class)
        llm_type (str): The type of llm, llama2, etc (Optional, defaults to "None") (Parent Class)
        task (str): The task (Optional, defaults to QA) (Parent Class)
        source (str): The source of the prompt (Optional, defaults to "NoSource") (Parent Class)
        doc_chain (str): The doc chain for the prompt ("stuff", "refine") (Optional, defaults to "stuff") (Parent Class)
        language (str): The language of the prompt (Optional, defaults to "en") (Parent Class)
        filepath (str): The filepath of the prompt (Optional) (Parent Class)
        input_keys (List[str]): The input keys for the prompt (Parent Class)
        input_keys (List[str]): The output keys for the prompt
        prefix (str): The template prefix for the prompt
        suffix (str): The template suffix for the prompt
        example_template (str): The template for formatting the examples
        examples (List[Dict[str, Any]]): The list of examples, each example is a dictionary with respective keys
    """

    output_keys: List[str]
    examples: List[Dict[str, Any]]
    prefix: str
    suffix: str
    example_template: str

    def __init__(self, **kwargs):
        """Initialize the prompt."""
        super().__init__(**kwargs)
        eg_formatter = PromptTemplate(
            input_vars=self.input_keys + self.output_keys,
            template=self.example_template,
        )
        self.prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=eg_formatter,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_keys,
        )

    @field_validator("output_keys")
    @classmethod
    def validate_output_keys(cls, v) -> List[str]:
        """Validate the output_keys field."""
        if v is None or v == []:
            raise ValueError("output_keys cannot be empty")
        return v

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, v) -> List[Dict[str, Any]]:
        """Validate the examples field."""
        if v is None or v == []:
            raise ValueError("examples cannot be empty")
        for eg in v:
            if not all(key in eg for key in cls.input_keys):
                raise ValueError(f"input key(s) not in example {eg}")
            if not all(key in eg for key in cls.output_keys):
                raise ValueError(f"output key(s) not in example {eg}")
        return v
