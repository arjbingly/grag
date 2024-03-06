import json
import os
from typing import List

from importlib_resources import files
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

import prompts
from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.utils import get_config

conf = get_config()


class BasicRAG:
    def __init__(self,
                 model_name=None,
                 doc_chain='refine',
                 task='QA',
                 llm_kwargs=None,
                 retriever_kwargs=None,
                 ):
        if retriever_kwargs is None:
            self.retriever = Retriever()
        else:
            self.retriever = Retriever(**retriever_kwargs)

        if llm_kwargs is None:
            self.llm_ = LLM()
        else:
            self.llm_ = LLM(**llm_kwargs)

        if model_name is None:
            self.llm = self.llm_.load_model()
            self.model_name = conf['llm']['model_name']
        else:
            self.model_name = model_name
            self.llm = self.llm_.load_model(model_name=self.model_name)

        _allowed_doc_chains = ['refine', 'stuff']
        if doc_chain not in _allowed_doc_chains:
            raise ValueError(f'Doc chain {doc_chain} is not allowed. Available choices: {_allowed_doc_chains}')
        self.doc_chain = doc_chain

        _allowed_tasks = ['QA']
        if task not in _allowed_tasks:
            raise ValueError(f'Task {task} is not allowed. Available tasks: {_allowed_tasks}')
        self.task = task

        self.main_prompt_name = f'{model_name}_{task}_1.json'
        self.refine_prompt_name = f'{model_name}_{task}-refine_1.json'
        self.prompt_path = files(prompts)

        self.main_prompt = self.load_prompt(self.prompt_path.joinpath(self.main_prompt_name))

        if doc_chain == 'refine':
            self.refine_prompt = self.load_prompt(self.prompt_path.joinpath(self.refine_prompt_name))

    def load_prompt(self, json_file: str | os.PathLike, return_input_vars=False):
        """
        Loads a prompt template from json file and returns a langchain ChatPromptTemplate

        Args:
            json_file: path to the prompt template json file.
            return_input_vars: if true returns a list of expected input variables for the prompt.

        Returns:
            langchain_core.prompts.ChatPromptTemplate (and a list of input vars if return_input_vars is True)

        """
        with open(f"{json_file}", "r") as f:
            prompt_json = json.load(f)
        prompt_template = ChatPromptTemplate.from_template(prompt_json['template'])

        input_vars = prompt_json['input_variables']

        return (prompt_template, input_vars) if return_input_vars else prompt_template

    @staticmethod
    def stuff_docs(docs: List[Document]) -> str:
        """
        Args:
            docs: List of langchain_core.documents.Document

        Returns:
            string of document page content joined by '\n\n'
        """
        return '\n\n'.join([doc.page_content for doc in docs])

    def stuff_call(self, query: str):
        retrieved_docs = self.retriever.get_chunk(query)
        context = self.stuff_docs(retrieved_docs)
        prompt = self.main_prompt.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        sources = [doc.metadata["source"] for doc in retrieved_docs]
        return response, sources

    def refine_call(self, query: str):
        retrieved_docs = self.retriever.get_chunk(query)
        sources = [doc.metadata["source"] for doc in retrieved_docs]
        responses = []
        for index, doc in enumerate(retrieved_docs):
            if index == 0:
                prompt = self.main_prompt.format(context=doc.page_content,
                                                 question=query)
                response = self.llm.invoke(prompt)
                responses.append(response)
            else:
                prompt = self.refine_prompt.format(context=doc.page_content,
                                                   question=query,
                                                   existing_answer=responses[-1])
                response = self.llm.invoke(prompt)
                responses.append(response)
        return responses, sources

    def __call__(self, query: str):
        if self.doc_chain == 'stuff':
            return self.stuff_call(query)
        elif self.doc_chain == 'refine':
            return self.refine_call(query)
