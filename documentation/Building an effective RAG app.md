# Building an effective RAG app

Source: [Building Production Ready RAG app: Jerry Liu](https://youtu.be/TRjq7t2Ms5I?si=mivlGwkQWr5g_iox)

**Note: Specifically for QA RAG**

## Challenges

### Retrieval Issues

1. **Low Precision**: Not all retrieved chunks are relevant.
 	This leads to hallucinations and being lost in middle problems
2. **Low Recall**: Not all relevant chunks are retrieved. This leads to the LLM not having enough context to generate the answer
3. **Outdated Information**: The data is redundant or out of date
 
### Generation Issues
 
1. **Hallucination**: LLM makes up an answer that is not in the provided context
2. **Irrelevance**: LLM makes up an answer that does not answer the question
3. **Toxicity/Bias**: LLM makes up an answer that is harmful or offensive

## Where can we Improve?

1. **Data**: Can we store additional information beyond raw text chunks?
Optimize data ingestion
2. **Embeddings**: Can we optimize our embedding representations?
Using pre-trained embeddings might not be optimal.
3. **Retrieval**: Can we do better than top-k embedding lookup?
4. **Synthesis**: Can we use LLMs for more than just generation? Like using LLM for reasoning, summarizing, breaking down questions, route questions, etc.

## Evaluation
**How do we evaluate a RAG system?**

1. Evaluate in isolation - retrieval, synthesis
2. Evaluate end-to-end

### Evaluating Retrieval
Evaluate the quality of retrieved chunks given a user query

1. Create an evaluation dataset
	- Input: Query
	- Output: the ground truth documents relevant to the query (id of docs).
2. Run the retriever over the dataset
3. Measure ranking metrics
	- Success rate
	- MRR
	- Hit-rate

*Evaluating synthesis alone is merely the performance of the LLM*

### Evaluating end-to-end
Evaluate the final generated response given input

1. Create an evaluation dataset
	- Input: Query
	- [Optional] Output: the ground truth answer
2. Run through the full RAG pipeline
3. Collect evaluation metrics
	- If no labels: label-free evals
	- If labels: with label evals

## Optimization

From simple (Less expensive, easier to implement, low time cost, low latency) to advanced (more expensive, harder to implement, high time cost, high latency) 

1. **Table Stakes**  (*Simple*)
	- Better parsers:

	 Making sure document parsers work properly, eg. pdf parsers handle tables well.
	 More retrieved tokens do not always equate to higher performance
	- Better chunking: eg. recursive chunker, chunk size
	- Hybrid search
	- Metadata filters

		Context you can inject into each text chunk. eg. document title, date of the document.
	
2. **Advanced Retrieval** (*Moderate*)
	- Reranking
	- Recursive retrieval
	- Embedding tables
	- Small to big retrieval

		Embed text at a small level and expand that window during LLM synthesis.

3. **Fine-tuning** (*Hard*)
	- Embedding fine-tuning
	- LLM fine-tuning

4. **Agentic Behavior**	 (*Hard*)
	- Routing
	- Query planning
	- Multi-document agents




 
 
 	   
 	
