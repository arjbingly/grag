from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from src.components.llm import LLM

var = (
    "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n\nAlways answer based only on the provided "
    "context. If the question can not be answered from the provided context, just say that you don't know, don't try to "
    "make up an answer.\n<</SYS>>\n\nUse the following pieces of context to answer the question at the end:\n\n\n{"
    "context}\n\n\nQuestion: {question}\n\nHelpful Answer: [/INST]"),

example_prompt = PromptTemplate(
    input_variables=["question", "answer", "statements"],
    template="Question: {question}\nAnswer: {answer}\nStatements: {statements}"
)

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

long_answer_template_prefix = '''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You have to create one or more statements from each sentence in the given answer
Always create statements from the given question and answer. If the question and answer does not have relevant statements, do not return anything, don't try to make up statements
<</SYS>>
Here are some examples:'''

long_answer_template_suffix = '''Create one or more statements from each sentence in the given answer:
Question: {question}
Answer: {answer}
Statements: [/INST]'''

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=long_answer_template_prefix,
    suffix=long_answer_template_suffix,
    input_variables=["question", "answer"],
)

llm_ = LLM()
llm = llm_.load_model()

prompt = prompt_template.format(question="Who was the father of Mary Ball Washington?",
                                answer="Roger, the school principal was the father of Mary Ball")
print(prompt)

llm.invoke(prompt)

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
# NLI_STATEMENTS_MESSAGE = Prompt(
#     name="nli_statements",
#     instruction="Natural language inference. Use only 'Yes' (1), 'No' (0) and 'Null' (-1) as verdict.",
#     examples=[
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
#     input_keys=["context", "statements"],
#     output_key="answer",
#     output_type="JSON",
# )
