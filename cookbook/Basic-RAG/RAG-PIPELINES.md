# RAG Pipelines

In very basic terms, a RAG pipeline is a system where relevant context is provided to the LLM along with the query. Typically, a vector database is leveraged for this task.

## Benefits of RAG

There are several advantages of using RAG:

1. Empowering LLM solutions with real-time data access
2. Preserving data privacy
3. Mitigating LLM hallucinations

# Considerations

## Document Chains

Document chains are chains used in RAG to effectively use retrieved documents. They are used for various purposes including efficient document processing, task decomposition, and improved accuracy.

### 1. Stuff Chain

This is the simplest document chain. This involves putting all relevant data into the prompt. Given 'n' documents, it just concatenated the documents with a separator usually, '\n\n'.

The advantage of this method is **it only requires one call to the LLM**, and the model has access to all the information at once.

However, one downside is **most LLMs can only handle a certain amount of context**. For large or multiple documents, stuffing may result in a prompt that exceeds the context limit.

Additionally, this method is **only suitable for smaller amounts of data**. When working with larger data, alternative approaches should be used.

![image](https://github.com/arjbingly/Capstone_5/assets/54805765/0b2bd6fa-4254-43ba-aa15-922a0f6ee8f0)  
[source](https://readmedium.com/en/https:/ogre51.medium.com/types-of-chains-in-langchain-823c8878c2e9)

### 2. Refine Chain

The Refine Documents Chain uses an iterative process to generate a response by analyzing each input document and updating its answer accordingly.

It passes all non-document inputs, the current document, and the latest intermediate answer to an LLM chain to obtain a new answer for each document.

This chain is ideal for tasks that involve analyzing more documents than can fit in the model’s context, as it **only passes a single document to the LLM at a time**.

However, this also means it makes significantly more LLM calls than other chains, such as the Stuff Documents Chain. It may **perform poorly for tasks that require cross-referencing between documents** or detailed information from multiple documents.

Pros of this method include **incorporating more relevant context and potentially less data loss** than the MapReduce Documents Chain. However, **it requires many more LLM calls and the calls are not independent, meaning they cannot be paralleled** like the MapReduce Documents Chain.

There may also be dependencies on the order in which the documents are analyzed. _Thus it might be ideal to provide documents in order of similarity. ??_

![image](https://github.com/arjbingly/Capstone_5/assets/54805765/bba655c5-954b-4a68-9317-3e272c7a543b)  
[source](https://readmedium.com/en/https:/ogre51.medium.com/types-of-chains-in-langchain-823c8878c2e9)

### 3. Map Reduce Chain

To process **large amounts of data efficiently**, the MapReduceDocumentsChain method is used.

This involves applying an LLM chain to each document individually (in the Map step), producing a new document. Then, all the new documents are passed to a separate combine documents chain to get a single output (in the Reduce step). If necessary, the mapped documents can be compressed before passing them to the combine documents chain.

This compression step is performed recursively.

This method requires an initial prompt on each chunk of data.

For summarization tasks, this could be a summary of that chunk, while for question-answering tasks, it could be an answer based solely on that chunk. Then, a different prompt is run to combine all the initial outputs.

The pros of this method are that **it can scale to larger documents and handle more documents** than the StuffDocumentsChain. Additionally, **the calls to the LLM on individual documents are independent and can be parallelized**.

The cons are that it **requires many more calls to the LLM** than the StuffDocumentsChain and **loses some information during the final combining call**.

![image](https://github.com/arjbingly/Capstone_5/assets/54805765/664c9a11-1d9a-4d85-8945-a43ee150708c)  
[source](https://readmedium.com/en/https:/ogre51.medium.com/types-of-chains-in-langchain-823c8878c2e9)

# Prompting

Prompting strategies differ from model to model.
eg. The Llama model takes system prompts

## Some Resources

1. [LangChain Smith Hub](https://smith.langchain.com)
2. [LangChain Hub](https://github.com/hwchase17/langchain-hub)

# Other Hyperparameters

- **Chunk Sizes** - generally, the smallest chunk size you can get away with.
- **Similarity score** - e.g. cosine similarity
- **Embedding**

# Sources

1. [RAG 101: Demystifying Retrieval-Augmented Generation Pipelines](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/)
2. [Mastering document chain in LangChain](https://www.comet.com/site/blog/mastering-document-chains-in-langchain/)
