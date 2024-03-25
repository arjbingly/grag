import streamlit as st

import os

# App title
st.set_page_config(page_title="RAG")
with st.sidebar:
    st.title('RAG ')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Llama2-7B', 'Llama2-13B', 'Mixtral 8x7B','Gemma 7B'], key='selected_model')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_k = st.sidebar.slider('top_k', min_value=1.0, max_value=5.0, value=1.0, step=1.0)


st.title("Welcome to the RAG App")

st.write(f"You have selected the {selected_model} model with the following parameters:")
st.write(f"Temperature: {temperature}")
st.write(f"Top-k: {top_k}")
# st.write(f"Max Length: {max_length}")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]



# def generate_response(user_input, model, temperature, top_k, max_length):
#     # inputs = tokenizer(user_input, return_tensors="pt")
#     outputs = model.generate(user_input, max_length=max_length, do_sample=True, top_k=top_k, temperature=temperature)
#     # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return outputs

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.text_area("Enter your query:", height=200)

# Process user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Here you can add the code to process the user input and generate a response using the selected model and parameters
    # For example:
    # response = generate_response(user_input, selected_model, temperature, top_k, max_length)
    # st.session_state.messages.append({"role": "assistant", "content": response})

