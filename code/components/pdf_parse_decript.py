from unstructured.partition.pdf import partition_pdf
import os
class PDF_parse:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def pdf_extract(self):
        """
        Parse and partition a PDF document into Text, Table and Image elements.
        
        Returns a dictionary with keys 'Text', 'Tables', and 'Images', each containing a list of their respective elements.
        
        """
        # Partition PDF file
        pdf_doc = partition_pdf(
            filename=self.pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            infer_table_structure=True,
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=None
        )
        
        # Initialize lists for each type of element
        classified_elements = {
            'Text': [],
            'Tables': [],
            'Images': []
        }
        
        # Check for element type, then append to appropriate list
        for element in pdf_doc:
            if element.category == "Table":
                classified_elements['Tables'].append(element)
            elif element.category == "Image":
                classified_elements['Images'].append(element)
            else:
                classified_elements['Text'].append(element)
        
        return classified_elements
#%%
# Test case
curr_dir = os.getcwd()
os.chdir('..')

pdf_path = 'test case data/pdf/llama2.pdf'
test_pdf = PDF_parse(pdf_path=pdf_path)
element_pdf = test_pdf.pdf_extract()
# %%
for element in element_pdf['Text']:
    print(element)
# %%
for element in element_pdf['Tables']:
    print(element)
# %%
for element in element_pdf['Images']:
    print(element)
# %%
