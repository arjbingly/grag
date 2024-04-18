"""A cookbook demonstrating how to run RAG app on streamlit."""

import os
import sys
from pathlib import Path

import streamlit as st
from grag.components.multivec_retriever import Retriever
from grag.components.utils import get_config
from grag.components.vectordb.deeplake_client import DeepLakeClient
from grag.rag.basic_rag import BasicRAG

sys.path.insert(1, str(Path(os.getcwd()).parents[1]))

st.set_page_config(page_title="GRAG",
                   menu_items={
                       "Get Help": "https://github.com/arjbingly/Capstone_5",
                       "About": "This is a simple GUI for GRAG"
                   })


def spinner(text):
    """Decorator that displays a loading spinner with a custom text message during the execution of a function.

    This decorator wraps any function to show a spinner using Streamlit's st.spinner during the function call,
    indicating that an operation is in progress. The spinner is displayed with a user-defined text message.

    Args:
        text (str): The message to display next to the spinner.

    Returns:
        function: A decorator that takes a function and wraps it in a spinner context.
    """

    def _spinner(func):
        """A decorator function that takes another function and wraps it to show a spinner during its execution.

        Args:
            func (function): The function to wrap.

        Returns:
            function: The wrapped function with a spinner displayed during its execution.
        """

        def wrapper_func(*args, **kwargs):
            """The wrapper function that actually executes the wrapped function within the spinner context.

            Args:
                *args: Positional arguments passed to the wrapped function.
                **kwargs: Keyword arguments passed to the wrapped function.
            """
            with st.spinner(text=text):
                func(*args, **kwargs)

        return wrapper_func

    return _spinner


@st.cache_data
def load_config():
    """Loads config."""
    return get_config()


conf = load_config()


class RAGApp:
    """Application class to manage a Retrieval-Augmented Generation (RAG) model interface.

    Attributes:
        app: The main application or server instance hosting the RAG model.
        conf: Configuration settings or parameters for the application.
    """

    def __init__(self, app, conf):
        """Initializes the RAGApp with a given application and configuration.

        Args:
            app: The main application or framework instance that this class will interact with.
            conf: A configuration object or dictionary containing settings for the application.
        """
        self.app = app
        self.conf = conf

    def render_sidebar(self):
        """Renders the sidebar in the application interface with model selection and parameters."""
        with st.sidebar:
            st.title('GRAG')
            st.subheader('Models and parameters')
            st.sidebar.selectbox('Choose a model',
                                 ['Llama-2-13b-chat', 'Llama-2-7b-chat',
                                  'Mixtral-8x7B-Instruct-v0.1', 'gemma-7b-it'],
                                 key='selected_model')
            st.sidebar.slider('Temperature',
                              min_value=0.1,
                              max_value=1.0,
                              value=0.1,
                              step=0.1,
                              key='temperature')
            st.sidebar.slider('Top-k',
                              min_value=1,
                              max_value=5,
                              value=3,
                              step=1,
                              key='top_k')
            st.button('Load Model', on_click=self.load_rag)
            st.checkbox('Show sources', key='show_sources')

    @spinner(text='Loading model...')
    def load_rag(self):
        """Loads the specified RAG model based on the user's selection and settings in the sidebar."""
        if 'rag' in st.session_state:
            del st.session_state['rag']

        llm_kwargs = {"temperature": st.session_state['temperature'], }
        if st.session_state['selected_model'] == "Mixtral-8x7B-Instruct-v0.1":
            llm_kwargs['n_gpu_layers'] = 16
            llm_kwargs['quantization'] = 'Q4_K_M'
        elif st.session_state['selected_model'] == "gemma-7b-it":
            llm_kwargs['n_gpu_layers'] = 18
            llm_kwargs['quantization'] = 'f16'

        retriever_kwargs = {
            "client_kwargs": {"read_only": True, },
            "top_k": st.session_state['top_k']
        }
        client = DeepLakeClient(collection_name="usc", read_only=True)
        retriever = Retriever(vectordb=client)

        st.session_state['rag'] = BasicRAG(model_name=st.session_state['selected_model'], stream=True,
                                           llm_kwargs=llm_kwargs, retriever=retriever,
                                           retriever_kwargs=retriever_kwargs)
        st.success(
            f"""Model Loaded !!!
    
    Model Name: {st.session_state['selected_model']}
    Temperature: {st.session_state['temperature']}
    Top-k     : {st.session_state['top_k']}"""
        )

    def clear_cache(self):
        """Clears the cached data within the application."""
        st.cache_data.clear()

    def render_main(self):
        """Renders the main chat interface for user interaction with the loaded RAG model."""
        st.title(":us: US Constitution Expert! :mortar_board:")
        if 'rag' not in st.session_state:
            st.warning("You have not loaded any model")
        else:
            user_input = st.chat_input("Ask me anything about the US Constitution.")

            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    _ = st.write_stream(
                        st.session_state['rag'](user_input)[0]
                    )
                    if st.session_state['show_sources']:
                        retrieved_docs = st.session_state['rag'].retriever.get_chunk(user_input)
                        for index, doc in enumerate(retrieved_docs):
                            with st.expander(f"Source {index + 1}"):
                                st.markdown(f"**{index + 1}. {doc.metadata['source']}**")
                                # if st.session_state['show_content']:
                                st.text(f"**{doc.page_content}**")

    def render(self):
        """Orchestrates the rendering of both main and sidebar components of the application."""
        self.render_main()
        self.render_sidebar()


if __name__ == "__main__":
    app = RAGApp(st, conf)
    app.render()
