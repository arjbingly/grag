from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from src.components.llm import LLM

# Faithfulness score = No of claims from the generated answer that can be inferred from the provided context /
#                       Total number of claims in the generated answer

# Let’s examine how faithfulness was calculated using the low faithfulness answer:
# Step 1: Break the generated answer into individual statements.
# Step 2: For each of the generated statements, verify if it can be inferred from the given context.
# Step 3: Use the formula depicted above to calculate faithfulness.


## Step 1:

# LONG_FORM_ANSWER_PROMPT = {
#     "name"= 'long_form_answer',
#     "instruction"= "Create one or more statements from each sentence in the given answer.",
#     "examples"=[
#         {
#             "question": "Who was  Albert Einstein and what is he best known for?",
#             "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
#             "statements": {
#                 "statements": [
#                     "Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.",
#                     "Albert Einstein was best known for his theory of relativity.",
#                     "Einstein's contributions significantly advanced the field of quantum mechanics",
#                     "Recognized globally, Einstein's work has profoundly impacted the scientific community",
#                     "Einstein's groundbreaking theories continue to shape our understanding of physics today.",
#                 ]
#             },
#         },
#         {
#             "question": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?",
#             "answer": "alcohol",
#             "statements": {
#                 "statements": ["Cadmium Chloride is slightly soluble in alcohol."]
#             },
#         },
#         {
#             "question": "Were Hitler and Benito Mussolini of the same nationality?",
#             "answer": "Sorry, I can't provide answer to that question.",
#             "statements": {"statements": []},
#         },
#     ]
# }
#
# %%
examples = [
    {
        "question": "Who was  Albert Einstein and what is he best known for?",
        "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
        "statements": {
            "statements": [
                "Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.",
                "Albert Einstein was best known for his theory of relativity.",
                "Einstein's contributions significantly advanced the field of quantum mechanics",
                "Recognized globally, Einstein's work has profoundly impacted the scientific community",
                "Einstein's groundbreaking theories continue to shape our understanding of physics today.",
            ]
        },
    },
    {
        "question": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?",
        "answer": "alcohol",
        "statements": {
            "statements": ["Cadmium Chloride is slightly soluble in alcohol."]
        },
    },
    {
        "question": "Were Hitler and Benito Mussolini of the same nationality?",
        "answer": "Sorry, I can't provide answer to that question.",
        "statements": {"statements": []},
    }]

for example in examples:
    statements = '\n'.join(f'{i}. {st}' for i, st in enumerate(example['statements']['statements'], 1))
    example['statements'] = statements
# %%
example_prompt_template = "Question: {question}\nAnswer: {answer}\nStatements: {statements}"
example_formatter = PromptTemplate(
    input_variables=["question", "answer", "statements"],
    template="Question: {question}\nAnswer: {answer}\nStatements: {statements}"
)

# long_answer_template_prefix = '''<s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. You have to create one or more statements from each sentence in the given answer
# Always create statements from the given question and answer. If the question and answer does not have relevant statements, do not return anything, don't try to make up statements.
# Create one or more statements from each sentence in the given answer as shown in the below examples.<</SYS>>
# '''

long_answer_template_prefix = '''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
Create one or more statements from each sentence in the given answer as shown in the below examples. Always create statements from the given question and answer. Do not make up anything.<</SYS>>
'''

long_answer_template_suffix = '''Create one or more statements from each sentence in the given answer:
Question: {question}
Answer: {answer}[/INST]
Statements: '''

long_answer_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_formatter,
    prefix=long_answer_template_prefix,
    suffix=long_answer_template_suffix,
    input_variables=["question", "answer"],
)
# %%
llm_ = LLM()
llm = llm_.load_model()
# %%
query = "Who was the father of Mary Ball Washington?"
gen_response = "The school principal, Roger was the father of Mary Ball"
prompt = long_answer_template.format(question=query,
                                     answer=gen_response)
print(prompt)

response = llm.invoke(prompt)
print(response)
# %%
query = "What unique ability does the newly discovered species of frog have?"
gen_response = "It can change its skin color based on the temperature of its environment"
prompt = long_answer_template.format(question=query,
                                     answer=gen_response)
print(prompt)

response = llm.invoke(prompt)
# print(response)
# %%
##Step 2

# NLI_STATEMENTS_MESSAGE = {
#     "name": "nli_statements",
#     "instruction": "Natural language inference. Use only 'Yes' (1), 'No' (0) and 'Null' (-1) as verdict.",
#     "examples": [
#         {
#             "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
#             "statements": """
#             statement_1: John is majoring in Biology.
#             statement_2: John is taking a course on Artificial Intelligence.
#             statement_3: John is a dedicated student.
#             statement_4: John has a part-time job.
#             """,
#             "answer": [
#                 {
#                     "statement_1": "John is majoring in Biology.",
#                     "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
#                     "verdict": "0",
#                 },
#                 {
#                     "statement_2": "John is taking a course on Artificial Intelligence.",
#                     "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
#                     "verdict": "0",
#                 },
#                 {
#                     "statement_3": "John is a dedicated student.",
#                     "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
#                     "verdict": "1",
#                 },
#                 {
#                     "statement_4": "John has a part-time job.",
#                     "reason": "There is no information given in the context about John having a part-time job.",
#                     "verdict": "0",
#                 },
#             ],
#         },
#         {
#             "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
#             "statements": """statement_1: Albert Einstein was a genius.""",
#             "answer": {
#                 "statement_1": "Albert Einstein was a genius.",
#                 "reason": "The context and statement are unrelated",
#                 "verdict": "0",
#             },
#         },
#         {
#             "context": """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.""",
#             "statements": """statement_1: Nil""",
#             "answer": {
#                 "statement_1": "Nil",
#                 "reason": "The statement is invalid",
#                 "verdict": "-1",
#             },
#         },
#     ],
#     "input_keys": ["context", "statements"],
#     "output_key": "answer",
#     "output_type": "JSON",
# }

# examples = [
#     {
#         "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
#         "statements": """
#             statement_1: John is majoring in Biology.
#             statement_2: John is taking a course on Artificial Intelligence.
#             statement_3: John is a dedicated student.
#             statement_4: John has a part-time job.
#             """,
#         "answer": [
#             {
#                 "statement_1": "John is majoring in Biology.",
#                 "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
#                 "verdict": "0",
#             },
#             {
#                 "statement_2": "John is taking a course on Artificial Intelligence.",
#                 "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
#                 "verdict": "0",
#             },
#             {
#                 "statement_3": "John is a dedicated student.",
#                 "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
#                 "verdict": "1",
#             },
#             {
#                 "statement_4": "John has a part-time job.",
#                 "reason": "There is no information given in the context about John having a part-time job.",
#                 "verdict": "0",
#             },
#         ],
#     },
#     {
#         "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
#         "statements": """statement_1: Albert Einstein was a genius.""",
#         "answer": [{
#             "statement_1": "Albert Einstein was a genius.",
#             "reason": "The context and statement are unrelated",
#             "verdict": "0",
#         }],
#     },
#     {
#         "context": """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.""",
#         "statements": """statement_1: Nil""",
#         "answer": [{
#             "statement_1": "Nil",
#             "reason": "The statement is invalid",
#             "verdict": "-1",
#         }],
#     },
# ]
#
# for example in examples:
#     ans = ''
#     for a in example["answer"]:
#         ans_ = '\n'.join(f'{key}: {val}' for key, val in a.items())
#         ans += f'\n{ans_}'
#     example["answer"] = ans
#
# example_formatter = PromptTemplate(
#     input_variables=["context", "statements", "answer"],
#     template="Context: {context}\nStatements: {statements}\n\nAnswer: {answer}"
# )
#
# nli_template_prefix = '''<s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. You have to do Natural language inference. Use only 'Yes' (1), 'No' (0) and 'Null' (-1) as verdict.
# <</SYS>>
# Here are some examples:'''
#
# nli_template_suffix = '''Natural language inference. Use only 'Yes' (1), 'No' (0) and 'Null' (-1) as verdict.:
# Context: {context}
# Statements: {statements}
# Answer: [/INST]'''
#
# nli_template = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_formatter,
#     prefix=nli_template_prefix,
#     suffix=nli_template_suffix,
#     input_variables=["context", "statements"],
# )
#
# prompt = nli_template.format(
#     context=gen_response,
#     statements='''statement_1: Mary Ball Washington's father was the school principal, Roger.
# statement_2: Roger was the father of Mary Ball Washington.'''
# )
#
# nli_response = llm.invoke(prompt)
