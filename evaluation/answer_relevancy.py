from pathlib import Path

import pandas as pd
import ragas
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness

from src.components.embedding import Embedding
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

emb = Embedding("instructor-embedding", "hkunlp/instructor-xl")
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

eval_df['answer'] = responses
dataset_dict = {
    "question": qna_df['Question'].to_list(),
    "answer": responses,
    "contexts": contexts,
    "ground_truth": qna_df['Answer'].to_list()
}

ds = Dataset.from_dict(dataset_dict)
# override the llm and embeddings for a specific metric
from ragas.metrics import answer_relevancy

answer_relevancy.llm = llm
answer_relevancy.embeddings = emb.embedding_function

# You can also init a new metric with the llm and embeddings of your choice

from ragas.metrics import AnswerRelevancy

ar = AnswerRelevancy(llm=llm, embeddings=emb.embedding_function)

# # pass to evaluate
result = ragas.evaluate(dataset=ds, metrics=[ar, answer_relevancy])
# even if I pass an llm or embeddings to evaluate, it will use the ones attached to the metrics
# result = ragas.evaluate(
#     metrics=[ar, answer_relevancy, faithfullness],
#     llm=llm,
#     embeddings=embeddings
# )
print(result)
