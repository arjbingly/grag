import streamlit as st

import os
from grag.rag.basic_rag import BasicRAG
from grag.components.utils import get_config
from grag.components.llm import LLM
conf = get_config()


# App title
st.set_page_config(page_title="RAG")

with st.sidebar:
    st.title('RAG ')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Llama2-7B','Llama2-13B','Mixtral 8x7B','Gemma 7B'], key='selected_model')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=1.0, max_value=5.0, value=1.0, step=1.0)


st.title("Welcome to the RAG App")


st.write(f"You have selected the {selected_model} model with the following parameters:")
st.write(f"Temperature: {temperature}")
st.write(f"Top-k: {top_k}")
# st.write(f"Max Length: {max_length}")
# rag = BasicRAG(model_name=selected_model, llm_kwargs={"temperature": temperature, "top_k": top_k})
llm = LLM(model_name=selected_model)
rag = BasicRAG(retriever=llm, model_name=selected_model)
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Add the initial prompt text to the messages
st.session_state.messages.append({"role": "assistant", "content": "How may I assist you today?"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.text_area("Enter your query:", height=200)

# Process user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response based on the selected model
    response, sources = rag(user_input)

    # Display the response
    for index, resp in enumerate(response):
        with st.chat_message("assistant"):
            st.write(f"Response {index + 1}: {resp}")
            st.write("Sources: ")
            for src_index, source in enumerate(sources[index]):
                st.write(f"\t{src_index}: {source}")
st.spinner("Creating the index...")
