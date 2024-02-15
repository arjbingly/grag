import numpy as np

import sys
import os
from pathlib import Path
sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from components.embedding import Embedding

#%%
def cosine_similarity(a,b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product/(magnitude_a * magnitude_b)

#%%
embedding_configs = [
    {'embedding_type' : 'sentence-transformers',
    'embedding_model' : "all-mpnet-base-v2",},
    {'embedding_type': 'instructor-embedding',
    'embedding_model': 'hkunlp/instructor-xl',}
]
# Test Documents
docs = ['Dynamic programming is both a mathematical optimization method and an algorithmic paradigm. The method was '
        'developed by Richard Bellman in the 1950s and has found applications in numerous fields, from aerospace '
        'engineering to economics.',
        'In computer science, recursion is a method of solving a computational problem where the solution depends on '
        'solutions to smaller instances of the same problem. Recursion solves such recursive problems by using '
        'functions that call themselves from within their own code. The approach can be applied to many types of '
        'problems, and recursion is one of the central ideas of computer science.']

print('Documents:')
for i, doc in enumerate(docs):
    print(f'{i+1}. {doc}')
    
query = 'What is recursion?'
print(f'Query: {query}')

# testing first config
print(f'Testing: {embedding_configs[0]}')

embedding = Embedding(**embedding_configs[0])
docs_vector = embedding.embedding_function.embed_documents(docs)
query_vector = embedding.embedding_function.embed_query(query)

similarity_scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in docs_vector]
print(f'Similairty Scores: {similarity_scores}')

if similarity_scores[1] > similarity_scores[0]:
    print('Test Passed')
else:
    print('Test Failed')

# testing second config
print(f'Testing: {embedding_configs[1]}')

embedding = Embedding(**embedding_configs[1])
docs_vector = embedding.embedding_function.embed_documents(docs)
query_vector = embedding.embedding_function.embed_query(query)

similarity_scores = [cosine_similarity(query_vector, doc_vector) for doc_vector in docs_vector]

print(f'Similairty Scores: {similarity_scores}')
if similarity_scores[1] > similarity_scores[0]:
    print('Test Passed')
else:
    print('Test Failed')

