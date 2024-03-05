# add code folder to sys path
import os
import sys
import time
from pathlib import Path

sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from src.components.parse_pdf import ParsePDF

from src.components.utils import get_config

data_path = Path(get_config()['data']['data_path'])
# %%
# data_path = data_path / 'test' / 'pdf'  # "data/test/pdf"
data_path = data_path / 'data' / 'pdf'


def main(filename):
    file_path = data_path / filename
    pdf_parser = ParsePDF()
    start_time = time.time()
    docs_dict = pdf_parser.load_file(file_path)
    time_taken = time.time() - start_time

    print(f'Parsed pdf in {time_taken:2f} secs')

    print(f'******** TEXT ********')
    for doc in docs_dict['Text']:
        print(doc)

    print(f'******** TABLES ********')
    for text_doc in docs_dict['Tables']:
        print(text_doc)

    print(f'******** IMAGES ********')
    for doc in docs_dict['Images']:
        print(doc)

    return docs_dict


if __name__ == "__main__":
    filename = '1040a.pdf'
    print(f'Parsing: {filename}')
    docs = main(filename)
    print('All Tests Passed')
