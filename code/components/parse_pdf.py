from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
import os


class ParsePDF:
    def __init__(self,
                 single_text_out=True,
                 strategy="hi_res",
                 infer_table_structure=True,
                 extract_images=True,
                 image_output_dir=None,
                 add_captions_to_text=True,
                 add_captions_to_blocks=True,
                 ):
        self.strategy = strategy
        if extract_images:
            self.extract_image_block_types = ["Image", "Table"]
        else:
            self.extract_image_block_types = ["Table"]
        self.infer_table_structure = infer_table_structure
        self.add_captions_to_text = add_captions_to_text
        self.add_captions_to_blocks = add_captions_to_blocks
        self.image_output_dir = image_output_dir
        self.single_text_out = single_text_out
        self.add_caption_first = True

    def partition(self, path):
        """
        Parse and partition a PDF document into elements.

        """
        self.file_path = path
        partitions = partition_pdf(
            filename=self.file_path,
            strategy=self.strategy,
            extract_images_in_pdf=True,  # Test if required
            extract_image_block_types=self.extract_image_block_types,
            infer_table_structure=self.infer_table_structure,
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self.image_output_dir
        )
        return partitions

    def classify(self, partitions):
        # Initialize lists for each type of element
        classified_elements = {
            'Text': [],
            'Tables': [],
            'Images': []
        }

        for i, element in enumerate(partitions):
            if element.category == "Table":
                if self.add_captions_to_blocks:
                    if partitions[i + 1].category == "FigureCaption":  # check for caption
                        caption_element = partitions[i + 1]
                    else:
                        caption_element = None
                    classified_elements['Tables'].append((element, caption_element))
                else:
                    classified_elements['Tables'].append((element, None))
            elif element.category == "Image":
                if self.add_captions_to_blocks:
                    if partitions[i + 1].category == "FigureCaption":  # check for caption
                        caption_element = partitions[i + 1]
                    else:
                        caption_element = None
                    classified_elements['Images'].append((element, caption_element))
                else:
                    classified_elements['Images'].append((element, None))
            else:
                if not self.add_captions_to_text:
                    if element.category != 'FigureCaption':
                        classified_elements['Text'].append(element)
                else:
                    classified_elements['Text'].append(element)

        return classified_elements

    def process_text(self, elements):
        if self.single_text_out:
            metadata = {'source': self.file_path} # Check for more metadata
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(page_content=text, metadata=metadata)]
        else:
            docs = []
            for element in elements:
                metadata = {'source': self.file_path,
                            'category': element.category}
                metadata.update(element.metadata.to_dict())
                docs.append(Document(page_content=str(element), metadata=metadata))
        return docs

    def process_tables(self, elements):
        docs = []
        for block_element, caption_element in elements:
            metadata = {'source': self.file_path,
                        'category': block_element.category}
            metadata.update(block_element.metadata.to_dict())
            if caption_element:
                if self.add_caption_first:
                    content = "\n\n".join([str(caption_element), str(block_element)])
                else:
                    content = "\n\n".join([str(block_element), str(caption_element)])
            else:
                content = str(block_element)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def process_images(self, elements):
        docs = []
        for block_element, caption_element in elements:
            metadata = {'source': self.file_path,
                        'category': block_element.category}
            metadata.update(block_element.metadata.to_dict())
            if caption_element:
                if self.add_caption_first:
                    content = "\n\n".join([str(caption_element), str(block_element)])
                else:
                    content = "\n\n".join([str(block_element), str(caption_element)])
            else:
                content = str(block_element)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def load_file(self, path):
        partitions = self.partition(path)
        classified_elements = self.classify(partitions)
        text_docs = self.process_text(classified_elements['Text'])
        table_docs = self.process_tables(classified_elements['Tables'])
        image_docs = self.process_tables(classified_elements['Images'])
        return {'Text': text_docs,
                'Tables': table_docs,
                'Images': image_docs}
# %%
