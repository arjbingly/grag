
# Capstone Proposal
## Creating an Retrival Augmented Generation based search for internal documents aided by Graph Neural Networks.
### Proposed by: Arjun Bingly
#### Email: arjbingly@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
The primary objective of this project is to develop an advanced search engine for internal documents, leveraging the capabilities of Language Models (LLMs) alongside Graph Neural Networks (GNNs) to enhance retrieval accuracy and relevance. The project aims to seamlessly handle diverse document formats, including unstructured (PDF, Markdown, Text, JSON) and structured (SQL, tables), by effectively integrating graph-based representations of document relationships.
            


## 2 Dataset:  

Since this project requires a diverse set of data the initial phase would use ArXiv data since we can not only obtain text-based data in various file formats like txt and pdf but also graph representations like _ogbn-arxiv_.  
 
            

## 3 Rationale:  

Most companies use a keyword-based search engine for internal documents, but the drawback of such an approach is that if the user does not use the specific keyword the search results are sub-par, thus using a vector-based search incorporates context-improving the results. Similarly, keyword-based search algorithms do not work well with long inquiries. Moreover, most of the readily available AI search engines published online do not use self-hosted services making them unusable for many industries. 

By incorporating graph-based representations and GNNs into this RAG pipeline, this project aims to advance the state-of-the-art in document search technology. The utilization of document relationships captured in graph structures will enable a more nuanced understanding of document context, ultimately improving the accuracy and relevance of document retrieval.
            

## 4 Approach:  

I plan on approaching this capstone through several steps.  

1. Literature review
2. Data sources and integration: like arXiv for text data, relational databases, etc.
3. Implement a basic RAG pipeline.
4. Improvements and incorporation of graphs and GNN into the pipeline

## 5 Timeline:  

This is a rough timeline for this project:  

- (3 Weeks) Literature review. 
- (3 Weeks) Data compiling  
- (3 Weeks) Pipeline coding and setup 
- (2 Weeks) Integrating various data sources  
- (1 Weeks) Mock Deployment  
- (1 Weeks) Writing Up a paper and submission
- (1 Weeks) Final Presentation  
            

## 6 Expected Number Students:  

4 enthusiastic students with backgrounds in NLP and LLMs.  
            

## 7 Possible Issues:  

The key challenges associated with the project would be:

- Integration of various kinds of data, eg. SQL databases, NO-SQL databases, etc.
- Implementation of a RAG pipeline.
- How can the data be enhanced using graphs? e.g. relational graphs, knowledge graphs.
- What methods can be used for a sub-graph retrieval system?
- Can GNN embeddings be an aid to the prompt?
- Can we avoid using vector databases and rely solely on graphs?
            

## Contact
- Author: Amir Jafari
- Email: [ajafari@gmail.com](Email)
- GitHub: [](Git Hub repo)
