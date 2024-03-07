# def generate_few_shot_prompt(task_description, examples, your_input):
#     """
#     Generates a few-shot prompt for LLMs like LLaMA-2 and Mixtral.
#
#     Args:
#         task_description (str): A brief description of the task.
#         examples (list of tuples): A list where each tuple contains an input text and its corresponding output.
#         your_input (str): The new input text you want the model to process based on the few-shot examples.
#
#     Returns:
#         str: A structured few-shot prompt ready to be inputted into an LLM.
#     """
#     prompt = f"Task: {task_description}\n\n"
#
#     for i, (input_text, output_text) in enumerate(examples, start=1):
#         prompt += f"Example {i}: \nInput: \"{input_text}\"\nOutput: \"{output_text}\"\n\n"
#
#     prompt += "Your Text: \"" + your_input + "\""
#
#     return prompt
#
#
# # Example usage
# task_description = "Summarize the following text in one sentence."
# examples = [
#     ("The quick brown fox jumps over the lazy dog.", "A fox leaps over a dog."),
#     ("LLaMA-2 and Mixtral are examples of large language models.", "LLaMA-2 and Mixtral are large language models.")
# ]
# your_input = "Few-shot prompting involves providing a model with a few examples to guide its output."
#
# prompt = generate_few_shot_prompt(task_description, examples, your_input)
# print(prompt)

# from langchain import FewShotPromptTemplate


few_shot_template = """

```json
{
"input": "Tk = T0 (ln k0/ ln k)"
"output": "import numpy as np; Tk = T0 * (np.log(k0) / np.log(k))"
}
```

```json
{
  "input": "Generate values from a Cauchy distribution with location parameter x0 and scale parameter gamma.",
  "output": "import numpy as np; cauchy_dist = np.random.standard_cauchy(size=1000) * gamma + x0"
}
```

```json
{
  "input": "Perform a Fast Fourier Transform (FFT) on a sequence of values.",
  "output": "import numpy as np; fft_result = np.fft.fft(sequence_of_values)"
}
```

```json
{
  "input": "Calculate the cross-entropy loss for predicted probabilities p and targets t.",
  "output": "import numpy as np; cross_entropy_loss = -np.sum(t * np.log(p + 1e-9)) / len(p)"
}
```

```json
{
  "input": "Compute the Total Harmonic Distortion (THD) given the harmonic levels (h) and the fundamental level (f) using THD = sqrt(sum(h^2)) / f.",
  "output": "import numpy as np; THD = np.sqrt(np.sum(np.array(h)**2)) / f"
}
```

"""