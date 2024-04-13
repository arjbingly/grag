"""Custom Few-Shot Prompts
============================
This cookbook demonstrates how to use custom few-shot prompts with Basic RAG.
"""

from grag.components.prompt import FewShotPrompt
from grag.rag.basic_rag import BasicRAG

custom_few_shot_prompt = FewShotPrompt(
    input_keys={"context", "question"},
    output_keys={"answer"},
    example_template="""
    question: {question}
    answer: {answer}
    """,
    prefix="""Answer the following question based on the given context like examples given below:""",
    suffix="""Answer the following question based on the given context
    question: {question}
    context: {context}
    answer:
    """,
    examples=[
        {
            "question": "What is the name of largest planet?",
            "answer": "Jupiter is the largest planet.",
        },
        {
            "question": "Who came up with Convolutional Neural Networks?",
            "answer": "Yann LeCun introduced convolutional neural networks.",
        },
    ],
)
rag = BasicRAG(doc_chain="stuff", custom_prompt=custom_few_shot_prompt)
