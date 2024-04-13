import numpy as np
import pytest
from grag.components.embedding import Embedding


# %%
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)


# %%
embedding_configs = [
    {
        "embedding_type": "sentence-transformers",
        "embedding_model": "all-mpnet-base-v2",
    },
    {
        "embedding_type": "instructor-embedding",
        "embedding_model": "hkunlp/instructor-xl",
    },
]


@pytest.mark.parametrize("embedding_config", embedding_configs)
def test_embeddings(embedding_config):
    # docs tuple format: (doc, similar to doc, asimilar to doc)
    doc_sets = [
        (
            "The new movie is awesome.",
            "The new movie is so great.",
            "The video is awful",
        ),
        (
            "The cat sits outside.",
            "The dog plays in the garden.",
            "The car is parked inside",
        ),
    ]
    embedding = Embedding(**embedding_config)
    for docs in doc_sets:
        doc_vecs = [embedding.embedding_function.embed_query(doc) for doc in docs]
        similarity_scores = [
            cosine_similarity(doc_vecs[0], doc_vecs[1]),
            cosine_similarity(doc_vecs[0], doc_vecs[2]),
        ]
    assert similarity_scores[0] > similarity_scores[1]
    del embedding
