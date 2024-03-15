# Evaluation metrics

## Traditional Measures

### Generation Metrics

1. **Perplexity** : *Fluency measure*
   Serves as an indicator of language processing efficacy. The lower the better. A model with a low perplexity score
   demonstrates high confidence and accuracy in its prediction, reflecting a strong understanding.

   $$ P = e^H $$
   where $P$ is the perplexity score and $H$ is the average cross-entropy, as shown below.
   $$H = - \frac{1}{N}\sum_{i=1}^{N}log_e (P(w_i|w_{j<i}))$$
   where $N$ is the total number of words and $P(w_i|w_{j<i})$ is the probability of the word $w_i$ given all the
   previous words $w_1, w_2, w_3, ..., w_{i-1}$.

2. **ROGUE** (Recall-oriented understudy for gisting evaluation): *Response Coherence measure* 	
   Usually used for evaluating text-summarization tasks. The value ranges from 0 to 1, the higher the better, indicating
   that the context and generated text have high similarity.  
   Common sub-types:
    - **ROUGE-1**: Measures the precision (recall, f1) of unigram overlap between generated and reference text.
    - **ROUGE-2**: Bigram
    - **ROUGE-N**: N-gram
    - **ROUGE-L**: Matches the Longest Common Sub-sequence (LCS).

---

### Retrieval Metrics

#### Order-Unaware

1. Recall
2. Precision
3. F1 Score

#### Order-Aware

4. Mean Average Precision (MAP)
5. Mean Normalized Discounted Cumulative Gain (Mean NDCG)
6. Mean Reciprocal Rank (MRR)

## What is .. ?

**Response Coherence**  
This metric evaluates how well the model generates coherent and logical responses that align with the context of the
conversation. It assesses the ability of the model to provide meaningful and contextually relevant answers.

**Fluency**
It measures how well the model's responses are structured, grammatically correct, and linguistically coherent. It
assesses the model's ability to generate smooth and natural-sounding language.

## Why are these not suited for RAG?

Due to the lack of effective metrics for RAG, (or until RAGAS, more later.) The traditional metrics were used. However,
this comes with many disadvantages, almost making these metrics useless in my opinion.

### Generation Metrics

- Generation metrics like **ROUGE** were made for summarization, and hence they look for semantic similarity of the
  input to the generation. This is not the case with RAG, though the model should not make up anything not in the
  provided context. It by principle should not just summarize. Therefore, this metric would always favor smaller
  chunking which will not be effective for more complex queries.
- Generation metrics like **Perplexity**, only look for confidence. What we have noticed is that since most RAG queries
  are questions many times the model is very confident even when there is nothing in the context, this is due to the
  inherent knowledge of the llm.
- The only effective use case for these metrics we found was to choose the LLM.

---

### Retrieval Metrics

- All traditional retrieval metrics require a predefined dataset. Therefore, they can not be used to tune parsing
  methods, chunk size, or even test production data.
- We also noticed a huge lack of datasets with tables and other challenging contexts.
- The only effective use case we found for these metrics was to choose the embedding model.

---

### Other concerns

- Though evaluating the generation and retrieval individually is necessary, there also needs to be an end-to-end
  evaluation metric.
- The idea of evaluating, is to use the results to make informed changes to the hyperparameters, but none of these metrics gives us anything to do that. 
- Due to these challenges, the most used (until now) production-ready metric are user feedback-based. eg. RAG Triad of metrics.

## The Game Changer - RAGAS
The newer and apparently currently production-used metrics or methodologies is RAGAS. This is a suite of metrics that leverage LLMs for reference-free evaluation and this is an active research topic.
The advantages of this approach are:

- Automated testing
- Can use production dataset
- Actionable insights

Possible downsides (though the RAGAS team and following research have tried to address these) 

- Inherent bias
- Need for a powerful LLM, or the need to instruction-tune smaller LLM.
- The prompt (In the case of RAGAS)

The fact that this is used in production tells me that though there are some problems with this approach it is certainly better than the traditional metrics and also good enough to deploy. 

---

### The Metrics
1. **Faithfulness**		
	It is a generation evaluation metric. It measures the **factual accuracy of the generated answer**. The number of correct statements from the given contexts is divided by the total number of statements in the generated answer. This metric uses the question, contexts, and the answer.	
2. **Answer relevance**		
	It is a generation evaluation metric. It measures **how relevant and to the point the generated answer is to the question**. This metric is computed using the question and the answer. For example, the answer “France is in Western Europe.” to the question “Where is France and what is its capital?” would achieve a low answer relevancy because it only answers half of the question.
3. **Context precision** 	
	It is a retrieval evaluation metric. It measures the **signal-to-noise ratio of the retrieved context**. This metric is computed using the question and the contexts
4. **Context recall**	
	It is a retrieval evaluation metric. It measures **if all the relevant information required to answer the question was retrieved**. This metric is computed based on the ground_truth *(this is the only metric in the framework that relies on human-annotated ground truth labels)* and the contexts.


All metrics are scaled from 0 to 1, the higher the better.

They also provide end-to-end evaluation metrics like answer semantic similarity and answer correctness.


#### How it is done?
1. **Faithfulness**	- *Uses: question, answer and context*	
	This is done in 2 steps. 
	- First, given a question and generated answer, Ragas uses an LLM to figure out the statements that the generated answer makes. This gives a list of statements whose validity we have we have to check. 
	- In step 2, given the list of statements and the context returned, Ragas uses an LLM to check if the statements provided are supported by the context. 
	- The number of correct statements is summed up and divided by the total number of statements in the generated answer to obtain the score for a given example.
2. **Answer relevance**	 - *Uses: question and answer*		
	For a given generated answer Ragas uses an LLM to find out the probable questions that the generated answer would be an answer to and computes similarity to the actual question asked.
3. **Context precision** - *Uses: question and context*		
	Given a question, Ragas calls LLM to figure out sentences from the retrieved context that are needed to answer the question. A ratio between the sentences required and the total sentences in the context gives you the score.
4. **Context recall** - *Uses: ground truth and context*	
	Ragas calculates this by using the provided ground_truth answer and using an LLM to check if each statement from it can be found in the retrieved context. If it is not found that means the retriever was not able to retrieve the information needed to support that statement.
	
#### Improvements from RAGAS
- The current implementation of RAGAS uses few-shot prompting and they have seen considerable improvement from it.
- Also RAGAS uses some trick where they call the LLM multiple times for a single query to ensure that there is no hallucination, and they seem to be utilizing the fact that LLMs are more confident when they are not hallucinating.
- In ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems, Saad-Falcon et al. found that training your own LLM evaluator can have a better performance than zero-shot prompting. 

-------

#### Sources
- [Evaluating RAG Applications with RAGAS - Medium article](https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a)
- [Evaluatin RAG Pipelines - Langchain article](https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/)
- [RAG Evaluation - Blog](https://weaviate.io/blog/rag-evaluation) - more reading to do.
- [RAGAS - paper](https://arxiv.org/pdf/2309.15217v1.pdf)
- [ARES - paper](https://arxiv.org/pdf/2311.09476.pdf)
