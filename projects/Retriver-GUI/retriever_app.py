import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(1, str(Path(os.getcwd()).parents[1]))

from grag.components.multivec_retriever import Retriever


class PageHome:
    def __init__(self, app):
        self.app = app

    def render_sidebar(self):
        with st.sidebar:
            st.session_state.metadata_toggle = st.toggle('Show Metadata')
            st.session_state.top_k = st.number_input('Show Top K',
                                                     min_value=0,
                                                     value=3,
                                                     step=1)

    def render_search_form(self):
        st.markdown("Enter query")
        with st.form("search_form"):
            st.session_state.query = st.text_input("Query:", value='What is Artificial Intelligence?')
            return st.form_submit_button("Search")

    def get_search_results(self, _query, _top_k):
        return self.app.retriever.get_chunk(_query,
                                            top_k=_top_k,
                                            with_score=True)

    def render_search_results(self):
        with st.spinner("Searching for similar chunks with :" + st.session_state.query):
            results = self.get_search_results(st.session_state.query, st.session_state.top_k)
            has_results = len(results) != 0
        if not has_results:
            return st.markdown("Could not find anything similar.")
        # st.write(results)
        for i, (result, score) in enumerate(results):
            with st.expander(f':bulb:**{i}** - Similiarity Score: {score:.3f}'):
                st.write(result.page_content)
                if st.session_state.metadata_toggle:
                    st.write(result.metadata)

    def check_connection(self):
        response = self.app.retriever.vectordb.test_connection()
        if response:
            return True
        else:
            return False

    def render_stats(self):
        st.write(f'''
        **Chroma Client Details:** \n
            Host Address    : {self.app.retriever.vectordb.host}:{self.app.retriever.vectordb.port} \n
            Collection Name : {self.app.retriever.vectordb.collection_name} \n
            Embeddings Type : {self.app.retriever.vectordb.embedding_type} \n
            Embeddings Model: {self.app.retriever.vectordb.embedding_model} \n
            Number of docs  : {self.app.retriever.vectordb.collection.count()} \n
        ''')
        if st.button('Check Connection'):
            response = self.app.retriever.vectordb.test_connection()
            if response:
                st.write(':green[Connection Active]')
            else:
                st.write(':red[Connection Lost]')

    def render(self):
        self.render_sidebar()
        tab1, tab2 = st.tabs(['Search', 'Details'])
        with tab1:
            submitted = self.render_search_form()
            if submitted:
                self.render_search_results()
        with tab2:
            self.render_stats()


class App:

    def __init__(self):
        self.retriever = Retriever()

    def render(self):
        st.title('Retriever App')
        PageHome(self).render()


if __name__ == "__main__":
    App().render()

# based on https://blog.streamlit.io/finding-your-look-alikes-with-semantic-search/
