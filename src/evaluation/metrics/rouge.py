from rouge_score import rouge_scorer

from projects.BasicRAG.BasicRAG_v1 import call_rag

query = 'What types of dependencies does dependence analysis identify in loop programs?'
response, source, context = call_rag(query)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True, split_summaries=True)
scores = scorer.score('\n'.join([query, context]), response)
print("\n")
print(scores)
