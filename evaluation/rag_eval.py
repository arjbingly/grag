from pathlib import Path

import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from src.components.llm import LLM
from src.components.multivec_retriever import Retriever
from src.components.utils import get_config

config_path = Path(__file__).parent / 'config.ini'
config = get_config(config_path)

metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

qna_df = pd.read_csv('./data/v1/qna_data.csv')
print(qna_df.head().to_string())

eval_df = pd.DataFrame()
eval_df['ground_truth'] = qna_df['Answer']
eval_df['query'] = qna_df['Question']

retriever = Retriever(top_k=3, **config['multivec_retriever'], chroma_kwargs=config['chroma'])
llm_ = LLM(**config['llm'])
llm = llm_.load_model()
prompt_name = 'Llama-2_QA_1.json'
prompt_path = Path(__file__).parent / 'prompts' / prompt_name
prompt_template = load_prompt(prompt_path)

contexts = []
for i, row in qna_df.iterrows():
    retrieved_docs = retriever.get_chunk(row['Answer'])
    contexts.append(retrieved_docs)

eval_df['context'] = contexts

responses = []
for i, row in eval_df.iterrows():
    prompt = prompt_template.format(context=row['context'], question=row['query'])
    response = llm.invoke(prompt)
    responses.append(response)

eval_df['response'] = responses

result_zephyr = evaluate(eval_df, metrics=metrics)

print(result_zephyr)

# if __name__ == "__main__":
#     query = 'What types of dependencies does dependence analysis identify in loop programs?'
# responses, sources = call_rag(query)
