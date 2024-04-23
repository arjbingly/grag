"""Class for Basic RAG.

This module provides:

- BasicRAG
"""

import json
from typing import List, Optional, Union

from grag import prompts
from grag.components.llm import LLM
from grag.components.multivec_retriever import Retriever
from grag.components.prompt import FewShotPrompt, Prompt
from grag.components.utils import get_config
from importlib_resources import files
from langchain_core.documents import Document

conf = get_config()


class BasicRAG:
    """Class for Basis RAG.

    Attributes:
        model_name (str): Name of the llm model
        doc_chain (str): Name of the document chain, ("stuff", "refine"), defaults to "stuff"
        task (str): Name of task, defaults to "QA"
        llm_kwargs (dict): Keyword arguments for LLM class
        retriever_kwargs (dict): Keyword arguments for Retriever class
        custom_prompt (Prompt): Prompt, defaults to None
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        model_name=None,
        doc_chain="stuff",
        task="QA",
        llm_kwargs=None,
        retriever_kwargs=None,
        stream: bool = False,
        custom_prompt: Union[Prompt, FewShotPrompt, None] = None,
    ):
        """Initialize BasicRAG."""
        if retriever is None:
            if retriever_kwargs is None:
                self.retriever = Retriever(client_kwargs={'read_only': True})
            else:
                self.retriever = Retriever(**retriever_kwargs)
        else:
            self.retriever = retriever

        if llm_kwargs is None:
            self.llm_ = LLM()
        else:
            self.llm_ = LLM(**llm_kwargs)

        self.prompt_path = files(prompts)
        self.custom_prompt = custom_prompt

        self._task = "QA"
        self.model_name = model_name
        self.doc_chain = doc_chain
        self.task = task
        self.stream = stream

        if self.custom_prompt is None:
            self.main_prompt = Prompt.load(
                self.prompt_path.joinpath(self.main_prompt_name)
            )

            if self.doc_chain == "refine":
                self.refine_prompt = Prompt.load(
                    self.prompt_path.joinpath(self.refine_prompt_name)
                )
        else:
            self.main_prompt = self.custom_prompt

    @property
    def model_name(self):
        """Return the name of the model."""
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        if value is None:
            self.llm = self.llm_.load_model()
            self._model_name = conf["llm"]["model_name"]
        else:
            self._model_name = value
            self.llm = self.llm_.load_model(model_name=self.model_name)

    @property
    def doc_chain(self):
        """Returns the doc_chain."""
        return self._doc_chain

    @doc_chain.setter
    def doc_chain(self, value):
        _allowed_doc_chains = ["refine", "stuff"]
        if value not in _allowed_doc_chains:
            raise ValueError(
                f"Doc chain {value} is not allowed. Available choices: {_allowed_doc_chains}"
            )
        self._doc_chain = value
        if value == "refine":
            if self.custom_prompt is not None:
                assert len(self.custom_prompt) == 2, ValueError(
                    f"Refine chain needs 2 custom prompts. {len(self.custom_prompt)} custom prompts were given."
                )
        self.prompt_matcher()

    @property
    def task(self):
        """Returns the task."""
        return self._task

    @task.setter
    def task(self, value):
        _allowed_tasks = ["QA"]
        if value not in _allowed_tasks:
            raise ValueError(
                f"Task {value} is not allowed. Available tasks: {_allowed_tasks}"
            )
        self._task = value
        self.prompt_matcher()

    def prompt_matcher(self):
        """Matches relvant prompt using model, task and doc_chain."""
        matcher_path = self.prompt_path.joinpath("matcher.json")
        with open(f"{matcher_path}", "r") as f:
            matcher_dict = json.load(f)
        try:
            self.model_type = matcher_dict[self.model_name]
        except KeyError:
            raise ValueError(
                f"Prompt for {self.model_name} not found in {matcher_path}"
            )

        self.main_prompt_name = f"{self.model_type}_{self.task}_1.json"
        self.refine_prompt_name = f"{self.model_type}_{self.task}-refine_1.json"
        if self.custom_prompt is None:
            self.main_prompt = Prompt.load(
                self.prompt_path.joinpath(self.main_prompt_name)
            )
            if self.doc_chain == "refine":
                self.refine_prompt = Prompt.load(
                    self.prompt_path.joinpath(self.refine_prompt_name)
                )

    @staticmethod
    def stuff_docs(docs: List[Document]) -> str:
        r"""Concatenates docs into a string seperated by '\n\n'.

        Args:
            docs: List of langchain_core.documents.Document

        Returns:
            string of document page content joined by '\n\n'
        """
        return "\n\n".join([doc.page_content for doc in docs])

    @staticmethod
    def output_parser(call_func):
        """Decorator to format llm output."""

        def output_parser_wrapper(*args, **kwargs):
            response, retrieved_docs = call_func(*args, **kwargs)
            if conf["llm"]["std_out"] == "False":
                # if self.llm_.callback_manager is None:
                print(response)
            print("Sources: ")
            for index, doc in enumerate(retrieved_docs):
                print(f"\t{index}: {doc.metadata['source']}")
            return response, retrieved_docs

        return output_parser_wrapper

    def stuff_chain(self, query: str):
        """Call function for stuff chain."""
        retrieved_docs = self.retriever.get_chunk(query)
        context = self.stuff_docs(retrieved_docs)
        prompt = self.main_prompt.format(context=context, question=query)
        return prompt, retrieved_docs

    @output_parser
    def stuff_call(self, query: str):
        """Call function for output of stuff chain."""
        prompt, retrieved_docs = self.stuff_chain(query)
        if self.stream:
            response = self.llm.stream(prompt)
        else:
            response = self.llm.invoke(prompt)
        return response, retrieved_docs

    def refine_chain(self, query: str):
        """Call function for refine chain."""
        retrieved_docs = self.retriever.get_chunk(query)
        responses = []
        for index, doc in enumerate(retrieved_docs[:-1]):
            if index == 0:
                prompt = self.main_prompt.format(
                    context=doc.page_content, question=query
                )
                response = self.llm.invoke(prompt)
                responses.append(response)
            else:
                prompt = self.refine_prompt.format(
                    context=doc.page_content,
                    question=query,
                    existing_answer=responses[-1],
                )
                response = self.llm.invoke(prompt)
                responses.append(response)
        prompt = self.refine_prompt.format(
            context=retrieved_docs[-1].page_content,
            question=query,
            existing_answer=responses[-1]
        )
        return prompt, retrieved_docs, responses

    @output_parser
    def refine_call(self, query: str):
        """Call function for output of refine chain."""
        prompt, retrieved_docs, responses = self.refine_chain(query)
        if self.stream:
            response = self.llm.stream(prompt)
        else:
            response = self.llm.invoke(prompt)
        responses.append(response)
        return responses, retrieved_docs

    def __call__(self, query: str):
        """Call function for the class."""
        if self.doc_chain == "stuff":
            return self.stuff_call(query)
        elif self.doc_chain == "refine":
            return self.refine_call(query)
