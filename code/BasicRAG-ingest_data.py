from pathlib import Path
import os
from tqdm import tqdm

from components.multivec_retriever import Retriever
from components.parse_pdf import  ParsePDF

data_path = Path(os.getcwd()).parent / 'data' / 'pdf'
# data_path = data_path / '0001'
formats_to_add = ['Text','Tables']
glob_pattern = '**/*.pdf'

parser = ParsePDF()
retriever = Retriever()

filepath_gen = data_path.glob(glob_pattern)
num_files = len(list(data_path.glob(glob_pattern)))
print(f'DATA PATH : {data_path}')
print(f'No of PDFs to add: {num_files}')
pbar = tqdm(filepath_gen, total=num_files, desc='Adding Files ')
for file in pbar:
    pbar.set_postfix({'Current file':file.relative_to(data_path)})
    docs = parser.load_file(file)
    for key in formats_to_add:
        retriever.add_docs(docs[key])
    pbar.write((f'Completed adding - {file.relative_to(data_path)}'))