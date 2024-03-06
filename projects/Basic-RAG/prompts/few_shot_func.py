from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

# create our examples
examples = [
    {
        "query": "Tk = T0 (ln k0/ ln k)",
        "answer": "import numpy as np; Tk = T0 * (np.log(k0) / np.log(k))"
    },
    {
        "query": "Generate values from a Cauchy distribution with location parameter x0 and scale parameter gamma.",
        "answer": "import numpy as np; cauchy_dist = np.random.standard_cauchy(size=1000) * gamma + x0"
    },
    {
        "query": "Perform a Fast Fourier Transform (FFT) on a sequence of values.",
        "answer": "import numpy as np; fft_result = np.fft.fft(sequence_of_values)"
    },
    {
        "query": "Calculate the cross-entropy loss for predicted probabilities p and targets t.",
        "answer": "import numpy as np; cross_entropy_loss = -np.sum(t * np.log(p + 1e-9)) / len(p)"
    },
    {
        "query": "Compute the Total Harmonic Distortion (THD) given the harmonic levels (h) and the fundamental level (f) using THD = sqrt(sum(h^2)) / f.",
        "answer": "import numpy as np; THD = np.sqrt(np.sum(np.array(h)**2)) / f"
    }
]

json_examples = [
    {
        "query": "FX July21 Call 120.5 / 125.0",
        "answer": """{{
            "trade": [
                {{
                    "entity": "FX",
                    "action": "Call",
                    "details": [
                        {{
                            "price": 120.5,
                            "quantity": 125.0,
                            "date": "July21"
                        }}
                    ]
                }}
            ]
        }}"""
    },
    {
        "query": "Commodity Sep22 Put 75.0 / 80.0",
        "answer": """{{
            "market": [
                {{
                    "entity": "Commodity",
                    "strategy": "Put",
                    "contracts": [
                        {{
                            "strike_price": 75.0,
                            "quantity": 80.0,
                            "expiry": "Sep22"
                        }}
                    ]
                }}
            ]
        }}"""
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """
[INST]
<<SYS>>
You are a helpful, respectful and honest assistant. You are good at writing python code.
You will be given with a formula or name of an equation or a concept. Based on your knowledge give the correct python
implementation of that as shown in below examples:
<</SYS>
"""

# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: 
[/INST]
"""

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=json_examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

query = "What is the formula for calculating F2 score, assume requirements."

print(few_shot_prompt_template.format(query=query))

# print(
#     "You are GPT-3, and you can't do math.\n\nYou can do basic math, and your memorization abilities are impressive, but you can't do any complex calculations that a human could not do in their head. You also have an annoying tendency to just make up highly specific, but wrong, answers.\n\nSo we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math problem, just use this format and we\u2019ll take care of the rest:\n\nQuestion: ${{Question with hard calculation.}}\n```python\n${{Code that prints what you need to know}}\n```\n```output\n${{Output of your code}}\n```\nAnswer: ${{Answer}}\n\nOtherwise, use this simpler format:\n\nQuestion: ${{Question without hard calculation}}\nAnswer: ${{Answer}}\n\nBegin.\n\nQuestion: What is 37593 * 67?\n\n```python\nprint(37593 * 67)\n```\n```output\n2518731\n```\nAnswer: 2518731\n\nQuestion: {question}\n")
