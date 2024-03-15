import json
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, field_validator, Field

Example = Dict[str, Any]

SUPPORTED_TASKS = ["QA"]
SUPPORTED_DOC_CHAINS = ["stuff", 'refine']


class Prompt(BaseModel):
    name: str = Field(default='Custom Prompt')
    llm_type: str
    task: str
    source: str = Field(default='NoSource')
    doc_chain: str
    language: str = 'en'
    filepath: Optional[str] = Field(default=None, exclude=True)
    input_keys: List[str]
    template: str
    prompt: Optional[PromptTemplate] = Field(exclude=True, repr=False, default=None)

    @field_validator("input_keys")
    @classmethod
    def validate_input_keys(cls, v) -> List[str]:
        if v is None or v == []:
            raise ValueError('input_keys cannot be empty')
        return v

    @field_validator("doc_chain")
    @classmethod
    def validate_doc_chain(cls, v: str) -> str:
        if v not in SUPPORTED_DOC_CHAINS:
            raise ValueError(
                f'The provided doc_chain, {v} is not supported, supported doc_chains are {SUPPORTED_DOC_CHAINS}')
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in SUPPORTED_TASKS:
            raise ValueError(f'The provided task, {v} is not supported, supported tasks are {SUPPORTED_TASKS}')
        return v

    # @model_validator(mode='after')
    # def load_template(self):
    #     self.prompt = ChatPromptTemplate.from_template(self.template)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = PromptTemplate(input_keys=self.input_keys, template=self.template)

    @classmethod
    def save(self, filepath: Union[Path, str, None], overwrite=False) -> Union[None, ValueError]:
        dump = self.model_dump_json(
            indent=2,
            exclude_defaults=True,
            exclude_none=True
        )
        if filepath is None:
            filepath = f'{self.name}.json'
        if overwrite:
            if self.filepath is None:
                return ValueError('filepath does not exist in instance')
            filepath = self.filepath
        with open(filepath, 'w') as f:
            f.write(dump)
        return None

    @classmethod
    def load(cls, filepath: Union[Path, str]):
        with open(f"{filepath}", "r") as f:
            prompt_json = json.load(f)
        _prompt = cls(**prompt_json)
        _prompt.filepath = str(filepath)
        return _prompt


class FewShotPrompt(Prompt):
    output_keys: List[str]
    examples: List[Dict[str, Any]]
    prefix: str
    suffix: str
    eg_template: str
    prompt: Optional[FewShotPromptTemplate] = Field(exclude=True, repr=False, default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        eg_formatter = PromptTemplate(input_vars=self.input_keys + self.output_keys,
                                      template=self.eg_template)
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
        if v is None or v == []:
            raise ValueError('output_keys cannot be empty')
        return v

    @field_validator('examples')
    @classmethod
    def validate_examples(cls, v) -> List[Dict[str, Any]]:
        if v is None or v == []:
            raise ValueError('examples cannot be empty')
        for eg in v:
            if not all(key in eg for key in cls.input_keys):
                raise ValueError(f"input key(s) not in example {eg}")
            if not all(key in eg for key in cls.output_keys):
                raise ValueError(f"output key(s) not in example {eg}")
        return v


if __name__ == '__main__':
    p = Prompt.load("../prompts/Llama-2_QA_1.json")
