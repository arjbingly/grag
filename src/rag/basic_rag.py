import json
import os
from typing import List

from importlib_resources import files
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.utils import get_config
from src.rag import prompts

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

        self.prompt_path = files(prompts)

        self._task = 'QA'
        self.model_name = model_name
        self.doc_chain = doc_chain
        self.task = task

        self.main_prompt = self.load_prompt(self.prompt_path.joinpath(self.main_prompt_name))

        if self.doc_chain == 'refine':
            self.refine_prompt = self.load_prompt(self.prompt_path.joinpath(self.refine_prompt_name))

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value is None:
            self.llm = self.llm_.load_model()
            self._model_name = conf['llm']['model_name']
        else:
            self._model_name = value
            self.llm = self.llm_.load_model(model_name=self.model_name)

    @property
    def doc_chain(self):
        return self._doc_chain

    @doc_chain.setter
    def doc_chain(self, value):
        _allowed_doc_chains = ['refine', 'stuff']
        if value not in _allowed_doc_chains:
            raise ValueError(f'Doc chain {value} is not allowed. Available choices: {_allowed_doc_chains}')
        self._doc_chain = value
        self.prompt_matcher()

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        _allowed_tasks = ['QA']
        if value not in _allowed_tasks:
            raise ValueError(f'Task {value} is not allowed. Available tasks: {_allowed_tasks}')
        self._task = value
        self.prompt_matcher()

    def prompt_matcher(self):
        matcher_path = self.prompt_path.joinpath('matcher.json')
        with open(f"{matcher_path}", "r") as f:
            matcher_dict = json.load(f)
        try:
            self.model_type = matcher_dict[self.model_name]
        except KeyError:
            raise ValueError(f'Prompt for {self.model_name} not found in {matcher_path}')

        self.main_prompt_name = f'{self.model_type}_{self.task}_1.json'
        self.refine_prompt_name = f'{self.model_type}_{self.task}-refine_1.json'
        self.main_prompt = self.load_prompt(self.prompt_path.joinpath(self.main_prompt_name))
        if self.doc_chain == 'refine':
            self.refine_prompt = self.load_prompt(self.prompt_path.joinpath(self.refine_prompt_name))

    @staticmethod
    def stuff_docs(docs: List[Document]) -> str:
        """
        Args:
            docs: List of langchain_core.documents.Document

        Returns:
            string of document page content joined by '\n\n'
        """
        return '\n\n'.join([doc.page_content for doc in docs])

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
    def output_parser(call_func):
        def output_parser_wrapper(*args, **kwargs):
            response, sources = call_func(*args, **kwargs)
            if conf['llm']['std_out'] == 'False':
                # if self.llm_.callback_manager is None:
                print(response)
            print(f'Sources: ')
            for index, source in enumerate(sources):
                print(f'\t{index}: {source}')

        return output_parser_wrapper

    @output_parser
    def stuff_call(self, query: str):
        retrieved_docs = self.retriever.get_chunk(query)
        context = self.stuff_docs(retrieved_docs)
        prompt = self.main_prompt.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        sources = [doc.metadata["source"] for doc in retrieved_docs]
        return response, sources

    @output_parser
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
