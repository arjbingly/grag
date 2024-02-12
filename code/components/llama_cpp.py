from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
# from InstructorEmbedding import INSTRUCTOR
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain import PromptTemplate, LLMChain

print(torch.cuda.is_available())
# template = """Question: {question}
#
# Answer: Let's work this out in a step by step way to be sure we have the right answer."""
#
# prompt = PromptTemplate.from_template(template)
#
# # Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# %%
loader = DirectoryLoader("./new_papers", glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
#                                                       model_kwargs={"device": "cuda"})
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
persist_directory = 'db'

## Here is the nmew embeddings being used
# embedding = embedding_function

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding_function,
                                 persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
# n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#
# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="llama-2-7b-chat.Q8_0.gguf",
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
# )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=False, )

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             use_auth_token=False,
                                             #  load_in_8bit=True,
                                             #  load_in_4bit=True
                                             )

# Use a pipeline for later
from transformers import pipeline

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})


# prompt = PromptTemplate(template=template, input_variables=["text"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)


qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)

query = "What is Flash attention?"
llm_response = qa_chain(query)
print(llm_response)
# process_llm_response(llm_response)
