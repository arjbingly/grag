from pathlib import Path

import pandas as pd
import ragas
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.utils import get_config, load_prompt

# os.environ["OPENAI_API_KEY"] = "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"

config_path = Path(__file__).parent / 'config.ini'
config = get_config(config_path)

# metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
metrics = [faithfulness, answer_relevancy, answer_correctness]

qna_df = pd.read_csv('./data/v1/qna_data.csv').iloc[:2]
# print(qna_df.head().to_string())

eval_df = pd.DataFrame()
eval_df['ground_truth'] = qna_df['Answer']
eval_df['question'] = qna_df['Question']

retriever = Retriever(top_k=3, **config['multivec_retriever'], chroma_kwargs=config['chroma'])
llm_ = LLM(**config['llm'])
llm = llm_.load_model()

prompt_name = 'Llama-2_QA_1.json'
prompt_path = Path(__file__).parent / 'prompts' / prompt_name
prompt_template = load_prompt(prompt_path)
print("------------- Model Loaded -------------")

contexts = []
for i, row in qna_df.iterrows():
    retrieved_docs = retriever.get_chunk(row['Answer'])
    contexts.append([doc.page_content for doc in retrieved_docs])

eval_df['context'] = contexts
print("------------- Context Added -------------")

responses = []
for i, row in eval_df.iterrows():
    prompt = prompt_template.format(context=row['context'], question=row['question'])
    response = llm.invoke(prompt)
    responses.append(response)

eval_df['response'] = responses
dataset_dict = {
    "question": qna_df['Question'].to_list(),
    "answer": responses,
    "contexts": contexts,
    "ground_truth": qna_df['Answer'].to_list()
}

ds = Dataset.from_dict(dataset_dict)
result_zephyr = ragas.evaluate(ds, metrics=metrics)

print(result_zephyr)

# examples = [
#     {"question": q, "ground_truth": a}
#     for i, (q, a) in enumerate(zip(qna_df['Question'].to_list(), qna_df['Answer'].to_list()))
# ]
# predictions = [
#     {"context": c, "response": r}
#     for i, (c, r) in enumerate(zip(contexts, responses))
# ]
# from ragas.langchain.evalchain import RagasEvaluatorChain
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )
#
# # create evaluation chains
# faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
# answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
# context_rel_chain = RagasEvaluatorChain(metric=context_precision)
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)
#
# res = faithfulness_chain.evaluate(examples, predictions)
# print(res)
# if __name__ == "__main__":
#     query = 'What types of dependencies does dependence analysis identify in loop programs?'
# responses, sources = call_rag(query)
