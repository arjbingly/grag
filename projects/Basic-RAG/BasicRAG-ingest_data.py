import os
from pathlib import Path
from uuid import UUID, uuid5

from tqdm import tqdm

from src.components.multivec_retriever import Retriever
from src.components.parse_pdf import ParsePDF
from src.components.utils import get_config

config = get_config()

DRY_RUN = False

data_path = Path(config['data']['data_path']) / 'pdf'
formats_to_add = ['Text', 'Tables']
glob_pattern = '**/*.pdf'

namespace = UUID('8c9040b0-b5cd-4d7c-bc2e-737da1b24ebf')
record_filename = uuid5(namespace, str(data_path))  # Unique file record name based on folder path

records_dir = Path(config['data']['data_path']) / 'records'
records_dir.mkdir(parents=True, exist_ok=True)
record_file = records_dir / f'{record_filename}.txt'


def load_processed_files():
    # Load processed files from a file if it exists
    if os.path.exists(record_file):
        with open(record_file, 'r') as file_record:
            processed_files.update(file_record.read().splitlines())


def update_processed_file_record(file_path, dry_run=False):
    # Update (append) the processed file record file
    with open(record_file, 'a') as file:
        if not dry_run:
            file.write(file_path + '\n')


def add_file_to_database(file_path: Path, dry_run=False):
    # Check if file_path is in the processed file set
    if str(file_path) not in processed_files:
        # Add file to the vector database
        add_to_database(file_path, dry_run=dry_run)
        # Add file_path to the processed file set
        processed_files.add(str(file_path))
        # Update the processed file record file
        update_processed_file_record(str(file_path), dry_run=dry_run)
        return f'Completed adding - {file_path.relative_to(data_path)}'
    else:
        return f'Already exists - {file_path.relative_to(data_path)}'


def add_to_database(file_path, dry_run=False):
    if not dry_run:
        docs = parser.load_file(file_path)
        for format_key in formats_to_add:
            retriever.add_docs(docs[format_key])


parser = ParsePDF()
retriever = Retriever()

processed_files = set()
load_processed_files()  # Load processed files into the set on script startup


def main():
    filepath_gen = data_path.glob(glob_pattern)
    num_files = len(list(data_path.glob(glob_pattern)))
    print(f'DATA PATH : {data_path}')
    print(f'No of PDFs to add: {num_files}')
    pbar = tqdm(filepath_gen, total=num_files, desc='Adding Files ')
    for file in pbar:
        pbar.set_postfix({'Current file': file.relative_to(data_path)})
        pbar.write(add_file_to_database(file, dry_run=DRY_RUN))
        # if str(file) not in processed_files:
        #     add_to_database(file, dry_run=DRY_RUN)  # Add file to the vector database
        #     processed_files.add(str(file))  # Add file_path to processed set
        #     update_processed_file_record(str(file), dry_run=DRY_RUN)  # Update the processed file record file
        #     pbar.write(f'Completed adding - {file.relative_to(data_path)}')
        # else:
        #     pbar.write(f'Already exists - {file.relative_to(data_path)}')


if __name__ == "__main__":
    main()
