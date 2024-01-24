
# Capstone Proposal
## Creating an AI-powered Search Engine for internal documents
### Proposed by: Arjun Bingly
#### Email: arjbingly@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
 
            The goal of this project is to develop a self-hosted grounded LLM-powered search engine based on concepts of vector 
            based similarity search and retrieval augmented generation for use by any company.  
            


## 2 Dataset:  

            The dataset is not finalized it would require more analysis of the feasibility but one approach would be to use arXiv
            papers as the data for the search engine but this would just be document-like data ideally, we should be able to 
            integrate structured databases as well since most data from the industry is in such a format. 
            

## 3 Rationale:  

            Most companies use a keyword-based search engine for internal documents, but the drawback of such an approach is 
            that if the user does not use the specific keyword the search results are sub-par, thus using a vector-based search
            incorporates context-improving the results. Similarly, keyword-based search algorithms do not work well with long
            inquiries. Moreover, most of the readily available AI search engines published online do not use self-hosted services
            making them unusable for many industries.
            

## 4 Approach:  

            I plan on approaching this capstone through several steps.  

            1. Literature review and work out the feasibility of the features.
            2. Data sources and integration: like arXiv for text data, relational databases, etc.
            3. Look at various approaches and methods: Vector database, knowledge graph, LLM, GNN, etc.  

## 5 Timeline:  

            This is a rough timeline for this project:  

            - (3 Weeks) Literature review and feasibility of features.  
            - (3 Weeks) Data compiling  
            - (3 Weeks) Pipeline coding and setup (Alpha)including the choice of similarity algorithm and model selection. 
            - (2 Weeks) Integrating various data sources  
            - (1 Weeks) Mock Deployment  
            - (1 Weeks) Writing Up a paper and submission
            - (1 Weeks) Final Presentation  
            

## 6 Expected Number Students:  

            4 enthusiastic students with backgrounds in NLP and LLMs.  
            

## 7 Possible Issues:  

            The key challenges associated with the project would be:
            i. To be able to run all services locally since many companies require that their sensitive data not leave their 
                premises.
            ii. Exploring newer techniques to make the process more efficient eg. using neural hashing to compress vectors for 
                search.
            iii. Exploring all the popular Approximate Nearest Neighbor algorithms like HNSW (Hierarchical Navigable Small World), 
                IVF (Inverted File), or PQ (Product Quantization) along with newer algorithms to find out the best performing for 
                the task.
            iv. Integration of various kinds of data, eg. SQL databases, NO-SQL databases, etc.
            


## Contact
- Author: Amir Jafari
- Email: [ajafari@gmail.com](Email)
- GitHub: [](Git Hub repo)
