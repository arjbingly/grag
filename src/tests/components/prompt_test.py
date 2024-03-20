from grag.components.prompt import Prompt
from importlib_resources import files

question = "What is the capital of France"
context = (
    "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 \
residents as of 1 January 2023 in an area of more than 105 km2 (41 sq mi), Paris is the fourth-most populated \
city in the European Union and the 30th most densely populated city in the world in 2022. Since the 17th century, \
Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy. \
For its leading role in the arts and sciences, as well as its early and extensive system of street lighting, in the \
19th century, it became known as the City of Light."
)


def test_prompt_files():
    prompt_files = list(files("grag.prompts").glob("*.json"))
    for file in prompt_files:
        if file.name.startswith("matcher"):
            prompt_files.remove(file)
    for file in prompt_files:
        prompt = Prompt.load(file)
        assert isinstance(prompt, Prompt)


def test_custom_prompt():
    template = """Answer the following question based on the given context.
        question: {question}
        context: {context}
        answer: 
        """
    correct_prompt = f"""Answer the following question based on the given context.
        question: {question}
        context: {context}
        answer: 
        """
    custom_prompt = Prompt(input_keys={"context", "question"}, template=template)
    assert custom_prompt.format(question=question, context=context) == correct_prompt
