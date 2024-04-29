"""Classes for parsing files.

This module provides:

- ParsePDF
"""

from grag.components.utils import configure_args
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf


@configure_args
class ParsePDF:
    """Parsing and partitioning PDF documents into Text, Table or Image elements.

    Attributes:
        single_text_out (bool): Whether to combine all text elements into a single output document.
        strategy (str): The strategy for PDF partitioning; default is "hi_res" for better accuracy.
        extract_image_block_types (list): Elements to be extracted as image blocks.
        infer_table_structure (bool): Whether to extract tables during partitioning.
        extract_images (bool): Whether to extract images.
        image_output_dir (str): Directory to save extracted images, if any.
        add_captions_to_text (bool): Whether to include figure captions in text output. Default is True.
        add_captions_to_blocks (bool): Whether to add captions to table and image blocks. Default is True.
        add_caption_first (bool): Whether to place captions before their corresponding image or table in the output. Default is True.
    """

    def __init__(
            self,
            single_text_out,
            strategy,
            infer_table_structure,
            extract_images,
            image_output_dir,
            add_captions_to_text,
            add_captions_to_blocks,
            table_as_html,
    ):
        """Initialize instance variables with parameters."""
        self.strategy = strategy
        if extract_images:  # by default always extract Table
            self.extract_image_block_types = [
                "Image",
                "Table",
            ]  # extracting Image and Table as image blocks
        else:
            self.extract_image_block_types = ["Table"]
        self.infer_table_structure = infer_table_structure
        self.add_captions_to_text = add_captions_to_text
        self.add_captions_to_blocks = add_captions_to_blocks
        self.image_output_dir = image_output_dir
        self.single_text_out = single_text_out
        self.add_caption_first = True
        self.table_as_html = table_as_html

    def partition(self, path: str):
        """Partitions a PDF document into elements based on the instance's configuration.

        Parameters:
            path (str): The file path of the PDF document to be parsed and partitioned.

        Returns:
            list: A list of partitioned elements from the PDF document.
        """
        self.file_path = path
        partitions = partition_pdf(
            filename=self.file_path,
            strategy=self.strategy,
            # extract_images_in_pdf=True,  # Not required if specifies extract_image_block_type
            extract_image_block_types=self.extract_image_block_types,
            infer_table_structure=self.infer_table_structure,
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self.image_output_dir,
        )
        return partitions

    def classify(self, partitions):
        """Classifies the partitioned elements into Text, Tables, and Images list in a dictionary.

        Also adds captions for each element (if available).

        Parameters:
            partitions (list): The list of partitioned elements from the PDF document.

        Returns:
            dict: A dictionary with keys 'Text', 'Tables', and 'Images', each containing a list of corresponding elements.
        """
        # Initialize lists for each type of element
        classified_elements = {"Text": [], "Tables": [], "Images": []}

        for i, element in enumerate(partitions):
            # enumerate, classify and add element + caption (when available) to respective list
            if element.category == "Table":
                if self.add_captions_to_blocks and i + 1 < len(partitions):
                    if (
                            partitions[i + 1].category == "FigureCaption"
                    ):  # check for caption
                        caption_element = partitions[i + 1]
                    else:
                        caption_element = None
                    classified_elements["Tables"].append((element, caption_element))
                else:
                    classified_elements["Tables"].append((element, None))
            elif element.category == "Image":
                if self.add_captions_to_blocks and i + 1 < len(partitions):
                    if (
                            partitions[i + 1].category == "FigureCaption"
                    ):  # check for caption
                        caption_element = partitions[i + 1]
                    else:
                        caption_element = None
                    classified_elements["Images"].append((element, caption_element))
                else:
                    classified_elements["Images"].append((element, None))
            else:
                if not self.add_captions_to_text:
                    if element.category != "FigureCaption":
                        classified_elements["Text"].append(element)
                else:
                    classified_elements["Text"].append(element)

        return classified_elements

    def text_concat(self, elements) -> str:
        """Context aware concatenates all elements into a single string."""
        full_text = ""
        for current_element, next_element in zip(elements, elements[1:]):
            curr_type = current_element.category
            next_type = next_element.category

            # if curr_type in ["FigureCaption", "NarrativeText", "Title", "Address", 'Table', "UncategorizedText", "Formula"]:
            #     full_text += str(current_element) + "\n\n"

            if curr_type == "Title" and next_type == "NarrativeText":
                full_text += str(current_element) + "\n"
            elif curr_type == "NarrativeText" and next_type == "NarrativeText":
                full_text += str(current_element) + "\n"
            elif curr_type == "ListItem":
                full_text += "- " + str(current_element) + "\n"
                if next_element == "Title":
                    full_text += "\n"
            elif next_element == "Title":
                full_text = str(current_element) + "\n\n"

            elif curr_type in ["Header", "Footer", "PageBreak"]:
                full_text += str(current_element) + "\n\n\n"

            else:
                full_text += "\n"

        return full_text

    def process_text(self, elements):
        """Processes text elements into langchain Documents.

        Parameters:
            elements (list): The list of text elements to be processed.

        Returns:
            docs (list): A list of Document instances containing the extracted Text content and their metadata.
        """
        if self.single_text_out:
            metadata = {"source": self.file_path}  # Check for more metadata
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(page_content=text, metadata=metadata)]
        else:
            docs = []
            for element in elements:
                metadata = {"source": self.file_path, "category": element.category}
                metadata.update(element.metadata.to_dict())
                docs.append(Document(page_content=str(element), metadata=metadata))
        return docs

    def process_tables(self, elements):
        """Processes table elements into Documents, including handling of captions if specified.

        Parameters:
            elements (list): The list of table elements (and optional captions) to be processed.

        Returns:
            docs (list): A list of Document instances containing Tables, their captions and metadata.
        """
        docs = []

        for block_element, caption_element in elements:
            metadata = {"source": self.file_path, "category": block_element.category}
            metadata.update(block_element.metadata.to_dict())
            if self.table_as_html:
                table_data = block_element.metadata.text_as_html
            else:
                table_data = str(block_element)

            if caption_element:
                if (
                        self.add_caption_first
                ):  # if there is a caption, add that before the element
                    content = "\n\n".join([str(caption_element), table_data])
                else:
                    content = "\n\n".join([table_data, str(caption_element)])
            else:
                content = table_data
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def process_images(self, elements):
        """Processes image elements into Documents, including handling of captions if specified.

        Parameters:
            elements (list): The list of image elements (and optional captions) to be processed.

        Returns:
            docs (list): A list of Document instances containing the Images, their caption and metadata.
        """
        docs = []
        for block_element, caption_element in elements:
            metadata = {"source": self.file_path, "category": block_element.category}
            metadata.update(block_element.metadata.to_dict())
            if caption_element:  # if there is a caption, add that before the element
                if self.add_caption_first:
                    content = "\n\n".join([str(caption_element), str(block_element)])
                else:
                    content = "\n\n".join([str(block_element), str(caption_element)])
            else:
                content = str(block_element)
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def load_file(self, path):
        """Loads a PDF file, partitions and classifies its elements, and processes these elements into Documents.

        Parameters:
            path (str): The file path of the PDF document to be loaded and processed.

        Returns:
            dict: A dictionary with keys 'Text', 'Tables', and 'Images', each containing a list of processed Document instances.
        """
        partitions = self.partition(str(path))
        classified_elements = self.classify(partitions)
        text_docs = self.process_text(classified_elements["Text"])
        table_docs = self.process_tables(classified_elements["Tables"])
        image_docs = self.process_images(classified_elements["Images"])
        return {"Text": text_docs, "Tables": table_docs, "Images": image_docs}
