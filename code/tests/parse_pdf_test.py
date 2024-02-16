import time

# add code folder to sys path
import os
from pathlib import Path
import sys
sys.path.insert(1, str(Path(os.getcwd()).parents[0]))

from components.parse_pdf import ParsePDF

#%%
data_path = Path(os.getcwd()).parents[1]/'data'/'test'/'pdf' #"data/test/pdf"
def main(filename):
    file_path = data_path/filename
    pdf_parser = ParsePDF()
    docs_dict = pdf_parser.load_file(file_path)
    print(f'******** TEXT ********')
    for doc in docs_dict['Text']:
        print(doc)

    print(f'******** TABLES ********')
    for text_doc in docs_dict['Tables']:
        print(text_doc)

    print(f'******** IMAGES ********')
    for doc in docs_dict['Images']:
        print(doc)

if __name__ == "__main__":
    filename = 'he_pdsw12.pdf'
    print(f'Parsing: {filename}')
    main(filename)
    print('All Tests Passed')