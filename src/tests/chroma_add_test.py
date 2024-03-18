import asyncio

# add code folder to sys path
import os
from pathlib import Path

from grag.components.chroma_client import ChromaClient

data_path = Path(os.getcwd()).parents[1] / 'data' / 'Gutenberg' / 'txt'  # "data/Gutenberg/txt"
docs = [
    '''And so on this rainbow day, with storms all around them, and blue sky
above, they rode only as far as the valley. But from there, before they
turned to go back, the monuments appeared close, and they loomed
grandly with the background of purple bank and creamy cloud and shafts
of golden lightning. They seemed like sentinels--guardians of a great
and beautiful love born under their lofty heights, in the lonely
silence of day, in the star-thrown shadow of night. They were like that
love. And they held Lucy and Slone, calling every day, giving a
nameless and tranquil content, binding them true to love, true to the
sage and the open, true to that wild upland home.''',
    '''Slone and Lucy never rode down so far as the stately monuments, though
these held memories as hauntingly sweet as others were poignantly
bitter. Lucy never rode the King again. But Slone rode him, learned to
love him. And Lucy did not race any more. When Slone tried to stir in
her the old spirit all the response he got was a wistful shake of head
or a laugh that hid the truth or an excuse that the strain on her
ankles from Joel Creech's lasso had never mended. The girl was
unutterably happy, but it was possible that she would never race a
horse again.''',
    '''Bostil wanted to be alone, to welcome the King, to lead him back to the
home corral, perhaps to hide from all eyes the change and the uplift
that would forever keep him from wronging another man.

The late rains came and like magic, in a few days, the sage grew green
and lustrous and fresh, the gray turning to purple.

Every morning the sun rose white and hot in a blue and cloudless sky.
And then soon the horizon line showed creamy clouds that rose and
spread and darkened. Every afternoon storms hung along the ramparts and
rainbows curved down beautiful and ethereal. The dim blackness of the
storm-clouds was split to the blinding zigzag of lightning, and the
thunder rolled and boomed, like the Colorado in flood.'''
]


def main():
    # print(f'Total Number of docs: {len(docs)}')

    client = ChromaClient(collection_name='test')

    print('Before Adding Docs...')
    print(f'The {client.collection_name} has {client.collection.count()} documents')

    n_docs = len(docs)
    print(f'Adding {n_docs} docs synchronously')
    client.add_docs(docs)
    print(f'Adding {n_docs} docs asynchronously')
    asyncio.run(client.aadd_docs(docs))

    print('After Adding Docs...')
    print(f'The {client.collection_name} has {client.collection.count()} documents')


if __name__ == "__main__":
    main()
    print('All Tests Passed')
