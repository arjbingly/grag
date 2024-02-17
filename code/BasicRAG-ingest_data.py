from pathlib import Path
import os

from components.multivec_retriever import Retriever
from components.parse_pdf import  ParsePDF

#%%
parser = ParsePDF()
retriever = Retriever()
#%%
data_path = Path(os.getcwd()).parent / 'data' / 'pdf'
# data_path = data_path / '0001'
formats_to_add = ['Text','Tables']
#%%
glob_pattern = '**/*.pdf'
filepath_gen = data_path.glob(glob_pattern)
num_files = len(list(data_path.glob(glob_pattern)))
print(f'No of PDFs to add: {num_files}')
#%%
print(f'DATA PATH : {data_path}')
#%%
for file in filepath_gen:
    print(f'Adding file: {file.relative_to(data_path)}')
    docs = parser.load_file(file)
    for key in formats_to_add:
        retriever.add_docs(docs[key])
