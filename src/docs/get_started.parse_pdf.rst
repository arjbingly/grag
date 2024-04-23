Parse PDF
=========

The parsing and partitioning were primarily done using the unstructured.io library, which is designed for this purpose. However, for PDFs with complex layouts, such as nested tables or tax forms, the pdfplumber and pytesseract libraries were employed to improve the parsing accuracy.

The class has several attributes that control the behavior of the parsing and partitioning process.

Attributes
##########

- single_text_out (bool): If True, all text elements are combined into a single output document. The default value is True.

- strategy (str): The strategy for PDF partitioning. The default is "hi_res" for better accuracy

- extract_image_block_types (list): A list of elements to be extracted as image blocks. By default, it includes "Image" and "Table".The default value is True.

- infer_table_structure (bool): Whether to extract tables during partitioning. The default value is True.

- extract_images (bool): Whether to extract images. The default value is True.

- image_output_dir (str): The directory to save extracted images, if any.

- add_captions_to_text (bool): Whether to include figure captions in the text output. The default value is True.

- add_captions_to_blocks (bool): Whether to add captions to table and image blocks. The default value is True.

- add_caption_first (bool): Whether to place captions before their corresponding image or table in the output. The default value is True.

- table_as_html (bool): Whether to represent tables as HTML.

Parsing Complex PDF Layouts
###########################

While unstructured.io performed well in parsing PDFs with straightforward layouts, PDFs with complex layouts, such as nested tables or tax forms, were not parsed accurately. To address this issue, the pdfplumber and pytesseract libraries were employed.

Table Parsing Methodology
=========================

For each page in the PDF file, the find_tables method is called with specific table settings to find the tables on that page. The table settings used are:

- ``"vertical_strategy": "text"``: This setting tells the function to detect tables based on the text content.

- ``"horizontal_strategy": "lines"``: This setting tells the function to detect tables based on the horizontal lines.

- ``"min_words_vertical": 3``: This setting specifies the minimum number of words required to consider a row as part of a table.

**For each table found on the page, the following steps are performed:**

1. The table area is cropped from the page using the crop method and the bbox (bounding box) of the table.

2. The text content of the cropped table area is extracted using the `extract_text` method with `layout=True`.

3. A dictionary is created with the `table_number` and `extracted_text` of the table, and it is appended to the `extracted_tables_in_page` list.
After processing all the tables on the page, a dictionary is created with the `page_number` and the list of `extracted_tables_in_page`, and it is appended to the `extracted_tables` list.
Finally, the extracted_tables list is returned, which contains all the extracted tables from the PDF file, organized by page and table number.

Limitations
===========

While the table parsing methodology using `pdfplumber` could process most tables, it could not parse every table layout accurately. The table settings need to be adjusted for different types of table layouts. Additionally, pdfplumber could not extract figure captions, whereas `unstructured.io` could.
Future work may involve developing a more robust and flexible table parsing algorithm that can handle a wider range of table layouts and integrate seamlessly with the ParsePDF class to leverage the strengths of both unstructured.io and pdfplumber libraries.
