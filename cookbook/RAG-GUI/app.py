import os
import sys
from pathlib import Path
import time
import streamlit as st

sys.path.insert(1, str(Path(os.getcwd()).parents[1]))
import shutil

st.set_page_config(page_title="RAG")
from grag.rag.basic_rag import BasicRAG
from grag.components.utils import get_config
from pathlib import Path
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
        self.exit_app = False
        if 'temperature' in st.session_state:
            self.temperature = st.session_state['temperature']
        else:
            self.temperature = 0.1

        if 'top_k' in st.session_state:
            self.top_k = st.session_state['top_k']
        else:
            self.top_k = 1.0
        self.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    def render_sidebar(self):
        with st.sidebar:
            st.title('RAG')
            st.subheader('Models and parameters')
            self.selected_model = st.sidebar.selectbox('Choose a model', ['Llama-2-13b-chat', 'Llama-2-7b-chat',
                                                                          'Mixtral-8x7B-Instruct-v0.1', 'Gemma 7B'],
                                                       key='selected_model')
            self.temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=self.temperature,
                                                 step=0.01)
            self.top_k = st.sidebar.slider('Top-k', min_value=1.0, max_value=5.0, value=self.top_k, step=1.0)
            st.session_state['temperature'] = self.temperature
            st.session_state['top_k'] = self.top_k

    def initialize_rag(self):
        llm_kwargs = {"temperature": self.temperature}
        retriever_kwargs = {
            "client_kwargs" : {"read_only": True,
                               "top_k": self.top_k}
        }
        rag = BasicRAG(model_name=self.selected_model, llm_kwargs=llm_kwargs,retriever_kwargs=retriever_kwargs)
        return rag

    def clear_cache(self):
        st.cache_data.clear()

    # @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    @st.cache(ignore_hash=True)
    def render_main(self):
        st.title("Welcome to the RAG App")

        st.write(f"You have selected the {self.selected_model} model with the following parameters:")
        st.write(f"Temperature: {self.temperature}")
        st.write(f"Top-k: {self.top_k}")

        if 'rag' not in st.session_state:
            st.session_state['rag'] = self.initialize_rag()

        for message in self.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.text_area("Enter your query:", height=20)
        submit_button = st.button("Submit")
        if submit_button and user_input:

                self.messages.append({"role": "user", "content": user_input})
                response, sources = st.session_state['rag'](user_input)
                st.write("LLM Output:")
                st.text_area(value=response)
                st.write("RAG Output:")
                # for index, resp in enumerate(rag_output):
                #     with st.expander(f"Response {index + 1}"):
                #         st.markdown(resp)
                #         st.write("Retrieved Chunks:")
                #         if isinstance(sources[index],(list,tuple)):
                #             for src_index, source in enumerate(sources[index]):
                #                 if hasattr(source, 'page_content'):
                #                     st.markdown(f"**Chunk {src_index + 1}:**\n{source.page_content}")
                #                 else:
                #                     st.markdown(f"**Chunk {src_index + 1}:**\n{source}")
                st.write("Response:")
                st.markdown(response)

                st.write("Sources:")
                # for index, source in enumerate(sources):
                #     st.write(f"{index + 1}. {source}")
    def render(self):


        self.clear_cache()
        self.render_sidebar()


        self.render_main()
        st.cache_data.clear()
        st.cache_resource.clear()
        self.exit_app = st.sidebar.button("Shut Down")




if __name__ == "__main__":
    lock_path = Path(conf['root']['root_path']) / '/data/vectordb/test/dataset_lock.lock'


    if os.path.exists(lock_path):
        shutil.rmtree(lock_path)
        print('Deleting lock file: {}'.format(lock_path))
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i + 1}')
        bar.progress(i + 1)
        time.sleep(0.1)


    app = RAGApp(st,conf)
    app.render()




