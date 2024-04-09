import os
import sys
from pathlib import Path

import streamlit as st
import tracemalloc
sys.path.insert(1, str(Path(os.getcwd()).parents[1]))

st.set_page_config(page_title="RAG")
from grag.rag.basic_rag import BasicRAG
from grag.components.utils import get_config
from grag.components.llm import LLM
from grag.components.multivec_retriever import Retriever
from grag.components.vectordb.deeplake_client import DeepLakeClient


@st.cache_resource
def load_config():
    return get_config()

conf = load_config()

class RAGApp:
    def __init__(self,app,conf):
        self.app = app
        self.conf = conf
        self.selected_model = None
        self.temperature = None
        self.top_k = None
        self.rag = None
        self.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        # if 'rag' in st.session_state:
        #     del st.session_state['rag']
        # self.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


    def render_sidebar(self):
        with st.sidebar:
            st.title('RAG')
            st.subheader('Models and parameters')
            self.selected_model = st.sidebar.selectbox('Choose a model', ['Llama-2-13b-chat', 'Llama-2-7b-chat', 'Mixtral-8x7B-Instruct-v0.1', 'Gemma 7B'], key='selected_model')
            self.temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
            self.top_k = st.sidebar.slider('Top-k', min_value=1.0, max_value=5.0, value=1.0, step=1.0)


    def initialize_rag(self):
        llm_kwargs = {"temperature": self.temperature}
        return BasicRAG(model_name=self.selected_model, llm_kwargs=llm_kwargs)


    def clear_cache(self):
        st.cache_data.clear()

    # @st.cache(suppress_st_warning=True, allow_output_mutation=True)

    def render_main(self):

        st.title("Welcome to the RAG App")

        st.write(f"You have selected the {self.selected_model} model with the following parameters:")
        st.write(f"Temperature: {self.temperature}")
        st.write(f"Top-k: {self.top_k}")

        if 'rag' not in st.session_state:
            st.session_state['rag'] = self.initialize_rag()

        for message in self.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_input = st.text_area("Enter your query:", height=20)
        submit_button = st.button("Submit")
        if submit_button and user_input:
            self.messages.append({"role": "user", "content": user_input})
            response, sources = st.session_state['rag'](user_input)

            # response, sources = st.session_state['rag'](user_input)

            for index, resp in enumerate(response):
                with st.chat_message("assistant"):
                    st.write(f"Response {index + 1}: {resp}")
                    st.write("Retrieved Chunks:")
                    for src_index, source in enumerate(sources[index]):
                        # st.write(f"\t{src_index + 1}: {source.page_content}")
                        if hasattr(source, 'page_content'):
                            st.write(f"\t{src_index + 1}: {source.page_content}")
                        else:
                            st.write(f"\t{src_index + 1}: {source}")


    def render(self):

        self.clear_cache()
        self.render_sidebar()


        self.render_main()




if __name__ == "__main__":
    tracemalloc.start()
    app = RAGApp(st,conf)
    app.render()

