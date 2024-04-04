import asyncio
import os
import shutil
from pathlib import Path

import pytest
from grag.components.utils import get_config
from grag.components.vectordb.deeplake_client import DeepLakeClient
from langchain_core.documents import Document

config = get_config()
test_path = Path(config['data']['data_path']) / 'vectordb/test_client'
if os.path.exists(test_path):
    shutil.rmtree(test_path)
    print('Deleting test retriever: {}'.format(test_path))


def test_deeplake_add_docs():
    docs = [
        """And so on this rainbow day, with storms all around them, and blue sky
    above, they rode only as far as the valley. But from there, before they
    turned to go back, the monuments appeared close, and they loomed
    grandly with the background of purple bank and creamy cloud and shafts
    of golden lightning. They seemed like sentinels--guardians of a great
    and beautiful love born under their lofty heights, in the lonely
    silence of day, in the star-thrown shadow of night. They were like that
    love. And they held Lucy and Slone, calling every day, giving a
    nameless and tranquil content, binding them true to love, true to the
    sage and the open, true to that wild upland home.""",
        """Slone and Lucy never rode down so far as the stately monuments, though
    these held memories as hauntingly sweet as others were poignantly
    bitter. Lucy never rode the King again. But Slone rode him, learned to
    love him. And Lucy did not race any more. When Slone tried to stir in
    her the old spirit all the response he got was a wistful shake of head
    or a laugh that hid the truth or an excuse that the strain on her
    ankles from Joel Creech's lasso had never mended. The girl was
    unutterably happy, but it was possible that she would never race a
    horse again.""",
        """Bostil wanted to be alone, to welcome the King, to lead him back to the
    home corral, perhaps to hide from all eyes the change and the uplift
    that would forever keep him from wronging another man.

    The late rains came and like magic, in a few days, the sage grew green
    and lustrous and fresh, the gray turning to purple.

    Every morning the sun rose white and hot in a blue and cloudless sky.
    And then soon the horizon line showed creamy clouds that rose and
    spread and darkened. Every afternoon storms hung along the ramparts and
    rainbows curved down beautiful and ethereal. The dim blackness of the
    storm-clouds was split to the blinding zigzag of lightning, and the
    thunder rolled and boomed, like the Colorado in flood.""",
    ]
    deeplake_client = DeepLakeClient(collection_name="test_client")
    if len(deeplake_client) > 0:
        deeplake_client.delete()
    docs = [Document(page_content=doc) for doc in docs]
    deeplake_client.add_docs(docs)
    assert len(deeplake_client) == len(docs)
    del deeplake_client


def test_deeplake_aadd_docs():
    docs = [
        """And so on this rainbow day, with storms all around them, and blue sky
    above, they rode only as far as the valley. But from there, before they
    turned to go back, the monuments appeared close, and they loomed
    grandly with the background of purple bank and creamy cloud and shafts
    of golden lightning. They seemed like sentinels--guardians of a great
    and beautiful love born under their lofty heights, in the lonely
    silence of day, in the star-thrown shadow of night. They were like that
    love. And they held Lucy and Slone, calling every day, giving a
    nameless and tranquil content, binding them true to love, true to the
    sage and the open, true to that wild upland home.""",
        """Slone and Lucy never rode down so far as the stately monuments, though
    these held memories as hauntingly sweet as others were poignantly
    bitter. Lucy never rode the King again. But Slone rode him, learned to
    love him. And Lucy did not race any more. When Slone tried to stir in
    her the old spirit all the response he got was a wistful shake of head
    or a laugh that hid the truth or an excuse that the strain on her
    ankles from Joel Creech's lasso had never mended. The girl was
    unutterably happy, but it was possible that she would never race a
    horse again.""",
        """Bostil wanted to be alone, to welcome the King, to lead him back to the
    home corral, perhaps to hide from all eyes the change and the uplift
    that would forever keep him from wronging another man.

    The late rains came and like magic, in a few days, the sage grew green
    and lustrous and fresh, the gray turning to purple.

    Every morning the sun rose white and hot in a blue and cloudless sky.
    And then soon the horizon line showed creamy clouds that rose and
    spread and darkened. Every afternoon storms hung along the ramparts and
    rainbows curved down beautiful and ethereal. The dim blackness of the
    storm-clouds was split to the blinding zigzag of lightning, and the
    thunder rolled and boomed, like the Colorado in flood.""",
    ]
    deeplake_client = DeepLakeClient(collection_name="test_client")
    if len(deeplake_client) > 0:
        deeplake_client.delete()
    docs = [Document(page_content=doc) for doc in docs]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(deeplake_client.aadd_docs(docs))
    assert len(deeplake_client) == len(docs)
    del deeplake_client


deeplake_get_chunk_params = [(1, False), (1, True), (2, False), (2, True)]


@pytest.mark.parametrize("top_k,with_score", deeplake_get_chunk_params)
def test_deeplake_get_chunk(top_k, with_score):
    query = """Slone and Lucy never rode down so far as the stately monuments, though
    these held memories as hauntingly sweet as others were poignantly
    bitter. Lucy never rode the King again. But Slone rode him, learned to
    love him. And Lucy did not race any more. When Slone tried to stir in
    her the old spirit all the response he got was a wistful shake of head
    or a laugh that hid the truth or an excuse that the strain on her
    ankles from Joel Creech's lasso had never mended. The girl was
    unutterably happy, but it was possible that she would never race a
    horse again."""
    deeplake_client = DeepLakeClient(collection_name="test_client", read_only=True)
    retrieved_chunks = deeplake_client.get_chunk(
        query=query, top_k=top_k, with_score=with_score
    )
    assert len(retrieved_chunks) == top_k
    if with_score:
        assert all(isinstance(doc[0], Document) for doc in retrieved_chunks)
        assert all(isinstance(doc[1], float) for doc in retrieved_chunks)
    else:
        assert all(isinstance(doc, Document) for doc in retrieved_chunks)
    del deeplake_client


@pytest.mark.parametrize("top_k,with_score", deeplake_get_chunk_params)
def test_deeplake_aget_chunk(top_k, with_score):
    query = """Slone and Lucy never rode down so far as the stately monuments, though
    these held memories as hauntingly sweet as others were poignantly
    bitter. Lucy never rode the King again. But Slone rode him, learned to
    love him. And Lucy did not race any more. When Slone tried to stir in
    her the old spirit all the response he got was a wistful shake of head
    or a laugh that hid the truth or an excuse that the strain on her
    ankles from Joel Creech's lasso had never mended. The girl was
    unutterably happy, but it was possible that she would never race a
    horse again."""
    deeplake_client = DeepLakeClient(collection_name="test_client", read_only=True)
    loop = asyncio.get_event_loop()
    retrieved_chunks = loop.run_until_complete(
        deeplake_client.aget_chunk(query=query, top_k=top_k, with_score=with_score)
    )
    assert len(retrieved_chunks) == top_k
    if with_score:
        assert all(isinstance(doc[0], Document) for doc in retrieved_chunks)
        assert all(isinstance(doc[1], float) for doc in retrieved_chunks)
    else:
        assert all(isinstance(doc, Document) for doc in retrieved_chunks)
    del deeplake_client
