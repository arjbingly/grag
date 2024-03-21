{
    "input_keys": ["context", "question"],
    "template": "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n\nAlways answer based only on the provided context. If the question can not be answered from the provided context, just say that you don't know, don't try to make up an answer.\n<</SYS>>\n\nUse the following pieces of context to answer the question at the end:\n\n\n{context}\n\n\nQuestion: {question}\n\nHelpful Answer: [/INST]",
    "llm_type": "Llama-2",
    "source": "https://github.com/hwchase17/langchain-hub/blob/master/prompts/vector_db_qa/prompt.json",
    "task": "QA",
}
