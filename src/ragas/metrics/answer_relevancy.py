# from langchain.prompts.few_shot import FewShotPromptTemplate
# from langchain.prompts.prompt import PromptTemplate
#
# from src.components.llm import LLM

examples = [
    {
        "answer": """Albert Einstein was born in Germany.""",
        "context": """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time""",
        "output": {
            "question": "Where was Albert Einstein born?",
            "noncommittal": 0,
        },
    },
    {
        "answer": """It can change its skin color based on the temperature of its environment.""",
        "context": """A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.""",
        "output": {
            "question": "What unique ability does the newly discovered species of frog have?",
            "noncommittal": 0,
        },
    },
    {
        "answer": """Everest""",
        "context": """The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas.""",
        "output": {
            "question": "What is the tallest mountain on Earth?",
            "noncommittal": 0,
        },
    },
    {
        "answer": """I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
        "context": """In 2023, a groundbreaking invention was announced: a smartphone with a battery life of one month, revolutionizing the way people use mobile technology.""",
        "output": {
            "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
            "noncommittal": 1,
        },
    },
]

for example in examples:
    statements = '\n'.join(f'{i}. {st}' for i, st in enumerate(example['statements']['statements'], 1))
    example['statements'] = statements

example_prompt = PromptTemplate(
    input_variables=["answer", "context", "output"],
    template="Answer: {answer}\nContext: {context}\nOutput: {output}"
)

long_answer_template_prefix = '''<s>[INST] <<SYS>>
Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer 
is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. 
For example, "I don't know" or "I'm not sure" are noncommittal answers.
<</SYS>>
Here are some examples:'''

long_answer_template_suffix = '''Create one or more statements from each sentence in the given answer:
Question: {question}
Answer: {answer}
Statements: [/INST]'''

long_answer_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=long_answer_template_prefix,
    suffix=long_answer_template_suffix,
    input_variables=["question", "answer"],
)

llm_ = LLM()
llm = llm_.load_model()

query = "Who was the father of Mary Ball Washington?"
gen_response = "The school principal, Roger was the father of Mary Ball"
prompt = long_answer_template.format(question=query,
                                     answer=gen_response)
print(prompt)

response = llm.invoke(prompt)
print(response)
